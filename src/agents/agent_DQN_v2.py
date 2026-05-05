"""
DQN V2 - Feature-extracted Double DQN for the Sailing Challenge.

Key design choices vs V1:
  - Replaces raw 49K CNN input with 70 physics-aware hand-crafted features.
  - This makes the network tiny (70→256→128→9), fast to train, and trivially
    exportable to pure-numpy for Codabench submission.
  - Double DQN update rule to reduce overestimation bias.
  - get_numpy_weights() extracts all params for submission generation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from agents.base_agent import BaseAgent


# ── Constants ──────────────────────────────────────────────────────────────────

GOAL        = np.array([64.0, 127.0])
GRID_SIZE   = 128
NUM_ACTIONS = 9
NUM_FEATURES = 70  # see extract_features()

ACTIONS_VEC = np.array([
    ( 0,  1), ( 1,  1), ( 1,  0), ( 1, -1),
    ( 0, -1), (-1, -1), (-1,  0), (-1,  1),
    ( 0,  0),
], dtype=np.float32)

# 8 compass directions (indices 0-7, same as ACTIONS_VEC rows 0-7)
_DIRS8 = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]


# ── Feature extraction ──────────────────────────────────────────────────────────

def compute_action_efficiencies(wind_vec: np.ndarray) -> np.ndarray:
    """Return the 9 sailing efficiencies for a given wind vector."""
    wspd = np.linalg.norm(wind_vec)
    if wspd < 1e-8:
        return np.full(9, 0.05, dtype=np.float32)
    wd = wind_vec / wspd
    effs = np.zeros(9, dtype=np.float32)
    for i, a in enumerate(ACTIONS_VEC):
        an = np.linalg.norm(a)
        if an < 1e-8:
            effs[i] = 0.0
            continue
        cos_a = float(np.clip(np.dot(a / an, wd), -1.0, 1.0))
        deg   = np.degrees(np.arccos(cos_a))
        if deg < 45:
            effs[i] = 0.05
        elif deg < 90:
            effs[i] = 0.5 + 0.5 * (deg - 45) / 45.0
        elif deg <= 135:
            effs[i] = 1.0
        else:
            effs[i] = 1.0 - 0.5 * (deg - 135) / 45.0
    return effs


def extract_features(obs: np.ndarray) -> np.ndarray:
    """
    Convert a raw 49158-dim sailing observation into 70 compact features.

    Layout (70 values):
      [0:3]   goal_dx/127, goal_dy/127, goal_dist/180     – goal direction
      [3:5]   x/127, y/127                                – absolute position
      [5:7]   unit-vector to goal (2 components)
      [7:10]  vx/8, vy/8, |v|/8                           – velocity
      [10:13] wx/15, wy/15, |w|/15                        – local wind
      [13:22] sailing efficiencies for each of 9 actions
      [22:46] obstacles in 8 dirs × 3 distances (4,10,20)
      [46:70] wind + obstacle along direct path (8 fracs × 3 values)
    """
    x, y   = float(obs[0]), float(obs[1])
    vx, vy = float(obs[2]), float(obs[3])
    wx, wy = float(obs[4]), float(obs[5])

    wind_field = obs[6          : 6 + 32768  ].reshape(128, 128, 2)
    world_map  = obs[6 + 32768  : 6 + 32768 + 16384].reshape(128, 128)

    ix = int(np.clip(x, 0, 127))
    iy = int(np.clip(y, 0, 127))

    gdx   = GOAL[0] - x
    gdy   = GOAL[1] - y
    gdist = float(np.sqrt(gdx**2 + gdy**2)) + 1e-8

    # (3) goal direction + distance
    f_goal = np.array([gdx / 127.0, gdy / 127.0, gdist / 180.0], dtype=np.float32)

    # (2) absolute position
    f_pos = np.array([x / 127.0, y / 127.0], dtype=np.float32)

    # (2) unit vector toward goal
    f_dir = np.array([gdx / gdist, gdy / gdist], dtype=np.float32)

    # (3) velocity
    vn = float(np.sqrt(vx**2 + vy**2))
    f_vel = np.array([vx / 8.0, vy / 8.0, vn / 8.0], dtype=np.float32)

    # (3) local wind
    wspd = float(np.sqrt(wx**2 + wy**2)) + 1e-8
    f_wind = np.array([wx / 15.0, wy / 15.0, wspd / 15.0], dtype=np.float32)

    # (9) action efficiencies
    f_eff = compute_action_efficiencies(np.array([wx, wy]))

    # (24) obstacles in 8 directions × 3 distances
    DISTS = [4, 10, 20]
    f_obs = np.zeros(24, dtype=np.float32)
    for k, (dx_o, dy_o) in enumerate(_DIRS8):
        for j, d in enumerate(DISTS):
            nx = int(np.clip(ix + dx_o * d, 0, 127))
            ny = int(np.clip(iy + dy_o * d, 0, 127))
            f_obs[k * 3 + j] = world_map[ny, nx]

    # (24) wind + obstacle at 8 points along the direct path to goal
    FRACS = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.00]
    f_path = np.zeros(24, dtype=np.float32)
    for i, frac in enumerate(FRACS):
        px = int(np.clip(x + gdx * frac, 0, 127))
        py = int(np.clip(y + gdy * frac, 0, 127))
        f_path[3 * i]     = world_map[py, px]
        f_path[3 * i + 1] = wind_field[py, px, 0] / 15.0
        f_path[3 * i + 2] = wind_field[py, px, 1] / 15.0

    return np.concatenate([f_goal, f_pos, f_dir, f_vel, f_wind, f_eff, f_obs, f_path])
    # 3+2+2+3+3+9+24+24 = 70


# ── Neural network ──────────────────────────────────────────────────────────────

class DQNNetworkV2(nn.Module):
    """70-dim → 256 → 128 → 9  (3-layer MLP, easily exported to numpy)."""

    def __init__(self, num_features: int = NUM_FEATURES, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Balanced replay buffer ──────────────────────────────────────────────────────

class BalancedReplayBuffer:
    """Equal capacity per training scenario so none dominates gradients."""

    def __init__(self, capacity_per_scenario: int, scenario_names: list):
        self.buffers        = {n: deque(maxlen=capacity_per_scenario) for n in scenario_names}
        self.scenario_names = scenario_names

    def push(self, feats, action, reward, next_feats, done, scenario):
        self.buffers[scenario].append((feats, action, reward, next_feats, done))

    def sample(self, batch_size: int):
        per     = batch_size // len(self.scenario_names)
        samples = []
        for n in self.scenario_names:
            buf = self.buffers[n]
            k   = min(per, len(buf))
            if k > 0:
                samples += random.sample(list(buf), k)
        random.shuffle(samples)
        s, a, r, ns, d = zip(*samples)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def ready(self, batch_size: int) -> bool:
        per = batch_size // len(self.scenario_names)
        return all(len(b) >= per for b in self.buffers.values())

    def __len__(self):
        return sum(len(b) for b in self.buffers.values())

    @property
    def total_capacity(self):
        return sum(b.maxlen for b in self.buffers.values())


# ── DQN Agent V2 ────────────────────────────────────────────────────────────────

class DQNAgentV2(BaseAgent):
    """
    Double DQN using 70-dim feature extraction.
    Designed so weights can be exported with get_numpy_weights() for
    pure-numpy Codabench submission (no torch required at inference).
    """

    def __init__(
        self,
        num_features:       int   = NUM_FEATURES,
        num_actions:        int   = NUM_ACTIONS,
        learning_rate:      float = 1e-3,
        discount_factor:    float = 0.995,
        exploration_rate:   float = 1.0,
        exploration_min:    float = 0.02,
        exploration_decay:  float = 0.9995,
        batch_size:         int   = 128,
        replay_capacity:    int   = 60_000,
        target_update_freq: int   = 300,
        device:             str   = None,
        checkpoint_path           = None,
    ):
        super().__init__()
        self.num_features       = num_features
        self.num_actions        = num_actions
        self.gamma              = discount_factor
        self.exploration_rate   = exploration_rate
        self.exploration_min    = exploration_min
        self.exploration_decay  = exploration_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.device             = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DQNNetworkV2(num_features, num_actions).to(self.device)
        self.target_net = DQNNetworkV2(num_features, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn     = nn.SmoothL1Loss()
        self.buffer      = BalancedReplayBuffer(
            replay_capacity // 3,
            ['training_1', 'training_2', 'training_3'],
        )
        self._step_count = 0

        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            if p.exists():
                self._load_weights(p)
            else:
                print(f"  No checkpoint at {p} — starting fresh")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_weights(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.exploration_rate = 0.0
        self.online_net.eval()
        print(f"  Loaded {path}  (exploration_rate set to 0)")

    def _feats_tensor(self, feats: np.ndarray) -> torch.Tensor:
        return torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def act(self, observation: np.ndarray) -> int:
        """Greedy action from raw observation (epsilon-greedy disabled — call act_train for training)."""
        feats = extract_features(observation)
        with torch.no_grad():
            return int(self.online_net(self._feats_tensor(feats)).argmax(dim=1).item())

    def reset(self) -> None:
        pass

    def seed(self, seed=None):
        super().seed(seed)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    # ── Training helpers ──────────────────────────────────────────────────────

    def act_train(self, feats: np.ndarray) -> int:
        """Epsilon-greedy action from pre-computed feature vector."""
        if np.random.rand() < self.exploration_rate:
            return int(np.random.randint(self.num_actions))
        with torch.no_grad():
            return int(self.online_net(self._feats_tensor(feats)).argmax(dim=1).item())

    def learn(self, feats, action, reward, next_feats, done, scenario: str):
        """Store transition (features, not raw obs) and update network."""
        self.buffer.push(feats, action, reward, next_feats, float(done), scenario)
        self._step_count += 1

        if not self.buffer.ready(self.batch_size):
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        st  = torch.tensor(s,  device=self.device)
        at  = torch.tensor(a,  device=self.device)
        rt  = torch.tensor(r,  device=self.device)
        nst = torch.tensor(ns, device=self.device)
        dt  = torch.tensor(d,  device=self.device)

        # Double DQN: online selects action, target evaluates value
        q_vals = self.online_net(st).gather(1, at.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a  = self.online_net(nst).argmax(dim=1, keepdim=True)
            next_q  = self.target_net(nst).gather(1, best_a).squeeze(1)
            targets = rt + self.gamma * next_q * (1.0 - dt)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_exploration(self):
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay,
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def get_numpy_weights(self) -> dict:
        """Return all network weights as numpy arrays (for submission generation)."""
        return {k: v.cpu().numpy() for k, v in self.online_net.state_dict().items()}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'step_count': self._step_count,
            'config': {
                'num_features':       self.num_features,
                'num_actions':        self.num_actions,
                'discount_factor':    self.gamma,
                'exploration_rate':   self.exploration_rate,
                'exploration_min':    self.exploration_min,
                'exploration_decay':  self.exploration_decay,
                'batch_size':         self.batch_size,
                'replay_capacity':    self.buffer.total_capacity,
                'target_update_freq': self.target_update_freq,
            },
        }, path)
        print(f"  Saved → {path}")

    @classmethod
    def load(cls, path, device=None, training_mode=False):
        ckpt  = torch.load(path, map_location=device or 'cpu', weights_only=False)
        agent = cls(device=device, **ckpt['config'])
        agent.online_net.load_state_dict(ckpt['online_net'])
        agent.target_net.load_state_dict(ckpt['target_net'])
        agent.optimizer.load_state_dict(ckpt['optimizer'])
        agent._step_count      = ckpt.get('step_count', 0)
        if training_mode:
            agent.online_net.train()
        else:
            agent.online_net.eval()
        return agent