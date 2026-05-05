# agent_PPO.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from agents.base_agent import BaseAgent


# ── Actor-Critic Network ────────────────────────────────────────────────────────

class PPONetwork(nn.Module):
    """
    Shared-backbone Actor-Critic network exploiting sailing observation structure:
        scalar    (6,)           → position, velocity, local wind
        wind CNN  (2, 128, 128)  → spatial wind patterns
        world CNN (1, 128, 128)  → island layout (static)

    Shared trunk → two heads:
        actor  → policy π(a|s)     logits over 9 actions
        critic → value  V(s)       scalar state value
    """
    def __init__(self, num_actions):
        super().__init__()

        # ── Branch 1: scalar state ────────────────────────────────────────────
        self.scalar_branch = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )  # → (64,)

        # ── Branch 2: wind field CNN ──────────────────────────────────────────
        self.wind_branch = nn.Sequential(
            nn.Conv2d(2, 8,  kernel_size=8, stride=4),   # → (8,  31, 31)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),   # → (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # → (32,  6,  6)
            nn.ReLU(),
            nn.Flatten(),                                 # → 1152
            nn.Linear(1152, 128),
            nn.ReLU(),
        )  # → (128,)

        # ── Branch 3: world map CNN ───────────────────────────────────────────
        self.world_branch = nn.Sequential(
            nn.Conv2d(1, 8,  kernel_size=8, stride=4),   # → (8,  31, 31)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),   # → (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),  # → (16,  6,  6)
            nn.ReLU(),
            nn.Flatten(),                                 # → 576
            nn.Linear(576, 64),
            nn.ReLU(),
        )  # → (64,)

        # ── Shared trunk: 64 + 128 + 64 = 256 ────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )  # → (64,)

        # ── Actor head: policy logits ─────────────────────────────────────────
        self.actor_head = nn.Linear(64, num_actions)   # → (num_actions,)

        # ── Critic head: state value ──────────────────────────────────────────
        self.critic_head = nn.Linear(64, 1)            # → (1,)

    def forward(self, x):
        scalar     = x[:,         :6            ]
        wind_flat  = x[:, 6       :6+32768      ]
        world_flat = x[:, 6+32768 :6+32768+16384]

        wind  = wind_flat.view(-1, 2, 128, 128)
        world = world_flat.view(-1, 1, 128, 128)

        s = self.scalar_branch(scalar)
        w = self.wind_branch(wind)
        m = self.world_branch(world)

        shared  = self.trunk(torch.cat([s, w, m], dim=1))
        logits  = self.actor_head(shared)               # policy logits
        value   = self.critic_head(shared).squeeze(-1)  # state value
        return logits, value

    def get_action(self, x):
        """Sample action + return log_prob and value for training."""
        logits, value = self.forward(x)
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        log_prob      = dist.log_prob(action)
        entropy       = dist.entropy()
        return action.item(), log_prob, value, entropy

    def evaluate(self, x, actions):
        """Evaluate stored actions — used during PPO update."""
        logits, value = self.forward(x)
        dist          = Categorical(logits=logits)
        log_probs     = dist.log_prob(actions)
        entropy       = dist.entropy()
        return log_probs, value, entropy


# ── Rollout Buffer ──────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores a single rollout (T steps) for PPO on-policy update.
    Cleared after each update — no long-term storage like DQN replay buffer.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions      = []
        self.log_probs    = []
        self.rewards      = []
        self.values       = []
        self.dones        = []
        self.scenarios    = []

    def push(self, obs, action, log_prob, reward, value, done, scenario):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.scenarios.append(scenario)

    def __len__(self):
        return len(self.rewards)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Generalised Advantage Estimation (GAE):
        A(t) = δ(t) + γλ·δ(t+1) + (γλ)²·δ(t+2) + ...
        where δ(t) = r(t) + γ·V(t+1) - V(t)   (TD error)

        GAE balances bias vs variance:
            λ=0 → pure TD (low variance, high bias)
            λ=1 → pure Monte Carlo (high variance, low bias)
            λ=0.95 → sweet spot used in original PPO paper
        """
        advantages = []
        returns    = []
        gae        = 0.0

        values = self.values + [last_value]   # append bootstrap value

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t]
                     + gamma * values[t + 1] * (1 - self.dones[t])
                     - values[t])
            gae   = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns


# ── PPO Agent ───────────────────────────────────────────────────────────────────

class PPOAgent(BaseAgent):
    def __init__(
        self,
        num_actions=9,
        learning_rate=3e-4,         # PPO typically uses higher lr than DQN
        discount_factor=0.995,
        gae_lambda=0.95,            # GAE smoothing parameter
        clip_epsilon=0.2,           # PPO clipping — key hyperparameter
        value_coef=0.5,             # weight of value loss
        entropy_coef=0.01,          # weight of entropy bonus (encourages exploration)
        max_grad_norm=0.5,
        n_epochs=4,                 # how many gradient steps per rollout
        rollout_steps=2048,         # steps collected before each update
        batch_size=64,
        device=None,
        checkpoint_path=Path(__file__).resolve().parent / "ppo_agent.pt",
    ):
        super().__init__()

        self.num_actions     = num_actions
        self.gamma           = discount_factor
        self.gae_lambda      = gae_lambda
        self.clip_epsilon    = clip_epsilon
        self.value_coef      = value_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm
        self.n_epochs        = n_epochs
        self.rollout_steps   = rollout_steps
        self.batch_size      = batch_size
        self.device          = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.network   = PPONetwork(num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.buffer    = RolloutBuffer()

        self._update_count = 0   # number of PPO updates performed

        # ── Auto-load checkpoint ──────────────────────────────────────────────
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            print(f"  Looking for checkpoint at: {checkpoint_path}")
            if checkpoint_path.exists():
                self._load_weights(checkpoint_path)
            else:
                print(f"  No checkpoint found — starting fresh")
        else:
            print(f"  No checkpoint path provided — starting fresh")

    def _load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network'])
        self.network.eval()
        print(f"  Weights loaded (inference mode)")

    def act(self, observation: np.ndarray) -> int:
        """
        Greedy action for evaluation (no exploration needed —
        PPO explores via the stochastic policy itself, not ε-greedy).
        """
        obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.network(obs_t)
            action    = logits.argmax(dim=1).item()   # greedy for eval
        return action

    def act_train(self, observation: np.ndarray):
        """
        Stochastic action for training — samples from policy distribution.
        Returns action, log_prob, value for storing in rollout buffer.
        """
        obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value, _ = self.network.get_action(obs_t)
        return action, log_prob.item(), value.item()

    def reset(self) -> None:
        pass

    def seed(self, seed=None) -> None:
        super().seed(seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def store(self, obs, action, log_prob, reward, value, done, scenario):
        """Store a single transition in the rollout buffer."""
        self.buffer.push(obs, action, log_prob, reward, value, done, scenario)

    def update(self, last_obs):
        """
        PPO update — called when rollout buffer is full (every rollout_steps).

        Key difference from DQN:
            DQN: update every step on random mini-batch from replay buffer
            PPO: collect T steps, update K times on those T steps, clear buffer
        """
        # Bootstrap value of last observation
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, last_value = self.network(last_obs_t)
            last_value    = last_value.item()

        # Compute GAE advantages and returns
        advantages, returns = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Convert buffer to tensors
        obs_t      = torch.tensor(
            np.array(self.buffer.observations), dtype=torch.float32, device=self.device)
        actions_t  = torch.tensor(
            self.buffer.actions, dtype=torch.long, device=self.device)
        old_lp_t   = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device)
        returns_t  = torch.tensor(
            returns, dtype=torch.float32, device=self.device)
        adv_t      = torch.tensor(
            advantages, dtype=torch.float32, device=self.device)

        # Normalise advantages (reduces variance, standard PPO practice)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T = len(self.buffer)
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        # ── K epochs over the same rollout data ───────────────────────────────
        for _ in range(self.n_epochs):
            # Shuffle indices for mini-batch updates
            indices = torch.randperm(T)

            for start in range(0, T, self.batch_size):
                end  = min(start + self.batch_size, T)
                idx  = indices[start:end]

                # Evaluate current policy on stored (s, a) pairs
                new_log_probs, values, entropy = self.network.evaluate(
                    obs_t[idx], actions_t[idx]
                )

                # ── PPO clipped surrogate loss ────────────────────────────────
                # ratio = π_new(a|s) / π_old(a|s)
                ratio        = torch.exp(new_log_probs - old_lp_t[idx])
                surr1        = ratio * adv_t[idx]
                surr2        = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * adv_t[idx]
                policy_loss  = -torch.min(surr1, surr2).mean()

                # ── Value loss ────────────────────────────────────────────────
                value_loss   = nn.functional.mse_loss(values, returns_t[idx])

                # ── Entropy bonus (encourages exploration) ────────────────────
                # Higher entropy = more random policy = more exploration
                entropy_loss = -entropy.mean()

                # ── Combined loss ─────────────────────────────────────────────
                loss = (policy_loss
                        + self.value_coef  * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()

        self._update_count += 1
        self.buffer.clear()   # ← on-policy: discard rollout after update

        n_batches = self.n_epochs * (T // self.batch_size + 1)
        return {
            'policy_loss': total_policy_loss / n_batches,
            'value_loss':  total_value_loss  / n_batches,
            'entropy':     total_entropy     / n_batches,
        }

    def save(self, path):
        torch.save({
            'network':   self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'num_actions':    self.num_actions,
                'learning_rate':  self.optimizer.param_groups[0]['lr'],
                'discount_factor':self.gamma,
                'gae_lambda':     self.gae_lambda,
                'clip_epsilon':   self.clip_epsilon,
                'value_coef':     self.value_coef,
                'entropy_coef':   self.entropy_coef,
                'max_grad_norm':  self.max_grad_norm,
                'n_epochs':       self.n_epochs,
                'rollout_steps':  self.rollout_steps,
                'batch_size':     self.batch_size,
            },
            'update_count': self._update_count,
        }, path)
        print(f"  Agent saved to '{path}'")

    @classmethod
    def load(cls, path, device=None, training_mode=False):
        checkpoint = torch.load(
            path, map_location=device or 'cpu', weights_only=False)
        config = checkpoint['config']
        agent  = cls(device=device, checkpoint_path=None, **config)
        agent.network.load_state_dict(checkpoint['network'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent._update_count = checkpoint['update_count']
        if training_mode:
            agent.network.train()
        else:
            agent.network.eval()
        return agent