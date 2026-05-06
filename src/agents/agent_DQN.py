import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from agents.base_agent import BaseAgent


# ── Balanced Replay Buffer ──────────────────────────────────────────────────────

class BalancedReplayBuffer:
    """Keeps equal replay capacity per scenario to prevent one dominating."""
    def __init__(self, capacity_per_scenario, scenario_names):
        self.buffers = {
            name: deque(maxlen=capacity_per_scenario)
            for name in scenario_names
        }
        self.scenario_names = scenario_names

    def push(self, state, action, reward, next_state, done, scenario):
        self.buffers[scenario].append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        per_scenario = batch_size // len(self.scenario_names)
        all_samples  = []
        for name in self.scenario_names:
            buf = self.buffers[name]
            if len(buf) >= per_scenario:
                all_samples += random.sample(list(buf), per_scenario)
            elif len(buf) > 0:
                all_samples += random.sample(list(buf), len(buf))
        random.shuffle(all_samples)
        states, actions, rewards, next_states, dones = zip(*all_samples)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return sum(len(b) for b in self.buffers.values())

    def ready(self, batch_size):
        per_scenario = batch_size // len(self.scenario_names)
        return all(len(b) >= per_scenario for b in self.buffers.values())

    @property
    def total_capacity(self):
        return sum(b.maxlen for b in self.buffers.values())


# ── Neural Network ──────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Multi-branch network exploiting the structure of the sailing observation:
        scalar    (6,)           → position, velocity, local wind
        wind CNN  (2, 128, 128)  → spatial wind patterns
        world CNN (1, 128, 128)  → island layout (static)
    Fusion head combines all branches → Q(s,a) for num_actions actions.
    """
    def __init__(self, num_actions):
        super().__init__()

        self.scalar_branch = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )  # → (64,)

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

        # 64 + 128 + 64 = 256
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        scalar     = x[:,         :6            ]   # (batch, 6)
        wind_flat  = x[:, 6       :6+32768      ]   # (batch, 32768)
        world_flat = x[:, 6+32768 :6+32768+16384]   # (batch, 16384)

        wind  = wind_flat.view(-1, 2, 128, 128)
        world = world_flat.view(-1, 1, 128, 128)

        s = self.scalar_branch(scalar)
        w = self.wind_branch(wind)
        m = self.world_branch(world)

        return self.fusion(torch.cat([s, w, m], dim=1))


# ── DQN Agent ───────────────────────────────────────────────────────────────────

class DQNAgent(BaseAgent):
    def __init__(
        self,
        num_actions=9,
        learning_rate=1e-4,
        discount_factor=0.995,
        exploration_rate=1.0,
        exploration_min=0.05,
        exploration_decay=0.9995,
        batch_size=64,
        replay_capacity=50_000,
        target_update_freq=500,
        device=None,
        checkpoint_path=Path(__file__).resolve().parent / "dqn_agent_early_stopping_balanced_buffer.pt",
    ):
        super().__init__()

        self.num_actions        = num_actions
        self.gamma              = discount_factor
        self.exploration_rate   = exploration_rate
        self.exploration_min    = exploration_min
        self.exploration_decay  = exploration_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DQNNetwork(num_actions).to(self.device)
        self.target_net = DQNNetwork(num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn     = nn.SmoothL1Loss()
        self.buffer      = BalancedReplayBuffer(
            capacity_per_scenario=replay_capacity // 3,
            scenario_names=['training_1', 'training_2', 'training_3']
        )
        self._step_count = 0

        # ── Auto-load checkpoint if it exists ────────────────────────────────
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
        """Load network weights and switch to inference mode."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.exploration_rate = 0.0
        self.online_net.eval()
        print(f"  Weights loaded — exploration rate set to 0.0 (inference mode)")

    def act(self, observation: np.ndarray) -> int:
        """Epsilon-greedy action selection. Satisfies BaseAgent interface."""
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return q_values.argmax(dim=1).item()

    def reset(self) -> None:
        """Buffer and weights persist across episodes — nothing to reset."""
        pass

    def seed(self, seed=None) -> None:
        """Extends BaseAgent.seed() with torch/random seeding."""
        super().seed(seed)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def learn(self, state, action, reward, next_state, done, scenario=None):
        """Store transition and update network if buffer is ready."""
        self.buffer.push(state, action, reward, next_state, done, scenario)  # ← scenario passed
        self._step_count += 1

        if not self.buffer.ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)

        q_values = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net picks action, target net evaluates it
            best_a  = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q  = self.target_net(next_states_t).gather(1, best_a).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_exploration(self):
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay
        )

    def get_numpy_weights(self) -> dict:
        """Return all network weights as numpy arrays (for numpy submission export)."""
        return {k: v.cpu().numpy() for k, v in self.online_net.state_dict().items()}

    def save(self, path):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'config': {
                'num_actions':        self.num_actions,
                'learning_rate':      self.optimizer.param_groups[0]['lr'],
                'discount_factor':    self.gamma,
                'exploration_rate':   self.exploration_rate,
                'exploration_min':    self.exploration_min,
                'exploration_decay':  self.exploration_decay,
                'batch_size':         self.batch_size,
                'replay_capacity':    self.buffer.total_capacity,   # ← fixed
                'target_update_freq': self.target_update_freq,
            },
            'step_count': self._step_count,
        }, path)
        print(f"  Agent saved to '{path}'")

    @classmethod
    def load(cls, path, device=None, training_mode=False):
        checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=False)
        config     = checkpoint['config']
        agent      = cls(device=device, **config)
        agent.online_net.load_state_dict(checkpoint['online_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent._step_count = checkpoint['step_count']
        if training_mode:
            agent.online_net.train()
        else:
            agent.online_net.eval()
        return agent