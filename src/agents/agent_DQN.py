import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ── Neural Network ──────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer ───────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ───────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        obs_dim,
        num_actions,
        hidden_dims=(128, 128),
        learning_rate=1e-3,
        discount_factor=0.995,
        exploration_rate=1.0,
        exploration_min=0.05,
        exploration_decay=0.998,
        batch_size=64,
        replay_capacity=50_000,
        target_update_freq=200,   # steps between target network syncs
        device=None,
    ):
        self.num_actions      = num_actions
        self.gamma            = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min  = exploration_min
        self.exploration_decay = exploration_decay
        self.batch_size       = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Online network (trained every step) + target network (synced periodically)
        self.online_net = DQNNetwork(obs_dim, num_actions, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(obs_dim, num_actions, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer(replay_capacity)
        self._step_count = 0

    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def act(self, observation):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        obs_t = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return q_values.argmax(dim=1).item()

    def learn(self, state, action, reward, next_state, done):
        """Store transition and train if buffer is ready."""
        self.buffer.push(state, action, reward, next_state, done)
        self._step_count += 1

        if len(self.buffer) < self.batch_size:
            return None   # not enough data yet

        # Sample minibatch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)

        # Current Q values for taken actions
        q_values = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Bellman)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically sync target network
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_exploration(self):
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay
        )

    def save(self, path):
        torch.save({
            'online_net':       self.online_net.state_dict(),
            'target_net':       self.target_net.state_dict(),
            'optimizer':        self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'step_count':       self._step_count,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.exploration_rate = checkpoint['exploration_rate']
        self._step_count      = checkpoint['step_count']