"""
Training script for DQN V2 (feature-extracted Double DQN).

Key improvements over V1:
  - Biased exploration: random actions are weighted toward goal-aligned,
    wind-efficient, obstacle-free directions (much faster early learning).
  - Stronger distance reward shaping (coeff 0.4 vs 0.01 in V1).
  - Features computed once per step (reused for next_feats, no double work).
  - Saves best model by success rate AND avg steps.
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from src.wind_scenarios import get_wind_scenario
from src.env_sailing import SailingEnv
from src.agents.agent_DQN_v2 import (
    DQNAgentV2,
    ACTIONS_VEC,
    extract_features,
)

SCENARIO_NAMES = ['training_1', 'training_2', 'training_3']
_DIRS8         = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]


# ── Biased exploration ──────────────────────────────────────────────────────────

def biased_random_action(feats: np.ndarray) -> int:
    """
    Sample a random action biased toward:
      - goal-aligned directions
      - wind-efficient directions
      - obstacle-free directions
    Much better than uniform random for sparse-reward navigation.
    """
    # Action efficiencies (index 13:22 in feature vector)
    effs = feats[13:22].copy()
    effs[8] = 0.0          # never 'stay' during random exploration

    # Goal direction alignment
    gdx = feats[0] * 127.0
    gdy = feats[1] * 127.0
    gdist = float(np.sqrt(gdx**2 + gdy**2)) + 1e-8

    align = np.array([
        max(0.0, (ACTIONS_VEC[i, 0] * gdx + ACTIONS_VEC[i, 1] * gdy) / gdist)
        for i in range(9)
    ], dtype=np.float32)
    align[8] = 0.0

    # Obstacle penalty: if obstacle at distance-4 in that direction → down-weight
    # Obstacle features start at index 22: 8 dirs × 3 distances; index 22+k*3 = near-distance for dir k
    obs_penalty = np.zeros(9, dtype=np.float32)
    for k in range(8):
        if feats[22 + k * 3] > 0.5:     # obstacle within 4 cells
            obs_penalty[k] = 3.0

    probs = 0.4 * effs + 0.6 * align - obs_penalty
    probs = np.maximum(probs, 1e-3)
    probs /= probs.sum()
    return int(np.random.choice(9, p=probs))


# ── Training loop ───────────────────────────────────────────────────────────────

def train_dqn_v2(
    agent:         DQNAgentV2,
    wind_scenarios: dict,
    num_episodes:  int,
    save_path:     str,
    biased_explore: bool = True,
):
    """Train DQN V2 on all wind scenarios with early stopping."""

    scenario_names = list(wind_scenarios.keys())
    n_sc           = len(scenario_names)

    rewards_h, steps_h, success_h = [], [], []
    loss_h, stuck_h, scenario_h   = [], [], []

    best_success  = 0.0
    best_steps    = float('inf')
    best_weights  = None
    no_improve    = 0
    patience      = 8
    best_per_sc   = {n: 0.0 for n in scenario_names}

    lr_reduced    = False
    decay_slowed  = False

    agent.seed(42)
    np.random.seed(42)

    t0 = time.time()
    print("=" * 65)
    print("  DQN V2 Training  (feature-extracted, numpy-exportable)")
    print(f"  Episodes        : {num_episodes}")
    print(f"  Scenarios       : {scenario_names}")
    print(f"  Device          : {agent.device}")
    print(f"  Biased explore  : {biased_explore}")
    print("=" * 65)

    for ep in range(num_episodes):
        sc_name = scenario_names[ep % n_sc]
        env     = SailingEnv(**wind_scenarios[sc_name])
        goal    = env.goal_position.copy()

        obs, info = env.reset(seed=ep)
        feats     = extract_features(obs)
        prev_dist = float(np.linalg.norm(info['position'] - goal))

        total_reward = 0.0
        losses       = []

        for step in range(500):
            # Action selection
            if np.random.rand() < agent.exploration_rate:
                action = biased_random_action(feats) if biased_explore \
                         else int(np.random.randint(9))
            else:
                action = agent.act_train(feats)

            next_obs, reward, done, truncated, info = env.step(action)
            next_feats = extract_features(next_obs)

            # Reward shaping: dense distance signal + island crash penalty
            curr_dist    = float(np.linalg.norm(info['position'] - goal))
            shaped_r     = reward + 0.4 * (prev_dist - curr_dist)
            if info.get('is_stuck', False):
                shaped_r = -50.0
            prev_dist = curr_dist

            loss = agent.learn(feats, action, shaped_r, next_feats,
                               done or truncated, sc_name)
            if loss is not None:
                losses.append(loss)

            feats        = next_feats
            obs          = next_obs
            total_reward += shaped_r

            if done or truncated:
                break

        env.close()

        is_stuck     = info.get('is_stuck', False)
        reached_goal = done and not is_stuck

        rewards_h.append(total_reward)
        steps_h.append(step + 1)
        success_h.append(reached_goal)
        stuck_h.append(is_stuck)
        loss_h.append(float(np.mean(losses)) if losses else 0.0)
        scenario_h.append(sc_name)

        # Adaptive exploration decay
        if agent.buffer.ready(agent.batch_size):
            recent_sr = sum(success_h[-100:]) / min(len(success_h), 100)

            if recent_sr >= 0.85 and not decay_slowed:
                agent.exploration_decay = 0.9999
                decay_slowed = True
                print(f"  Exploration decay slowed  (success={recent_sr*100:.1f}%)", flush=True)

            if recent_sr < 1.0:
                agent.decay_exploration()

        status = "✓" if reached_goal else ("~" if is_stuck else "✗")
        print(f"  [{status}] Ep {ep+1:>4} ({sc_name}) | "
              f"R={total_reward:+7.1f} | steps={step+1:>3} | "
              f"ε={agent.exploration_rate:.3f}",
              flush=True)

        # ── Every 100 episodes: summary + checkpoint ──────────────────────────
        if (ep + 1) % 100 == 0:
            sl   = success_h[-100:]
            stl  = stuck_h[-100:]
            curr_sr    = sum(sl) / len(sl)
            curr_stuck = sum(stl) / len(stl)
            curr_steps = float(np.mean(steps_h[-100:]))
            elapsed    = time.time() - t0
            eta        = (num_episodes - ep - 1) / ((ep + 1) / elapsed)

            print()
            print(f"  ── Ep {ep+1}/{num_episodes} | {elapsed:.0f}s | ETA {eta:.0f}s")
            print(f"     Success  (last 100): {curr_sr*100:.1f}%   "
                  f"Stuck: {curr_stuck*100:.1f}%")
            print(f"     Avg steps          : {curr_steps:.1f}")
            print(f"     Avg loss           : {np.mean(loss_h[-100:]):.5f}")
            print(f"     Exploration ε      : {agent.exploration_rate:.4f}")
            print(f"     LR                 : {agent.optimizer.param_groups[0]['lr']:.2e}")

            # LR reduction at 85% success
            if curr_sr >= 0.85 and not lr_reduced:
                old_lr = agent.optimizer.param_groups[0]['lr']
                for pg in agent.optimizer.param_groups:
                    pg['lr'] *= 0.2
                lr_reduced = True
                print(f"  ↓ LR {old_lr:.2e} → {agent.optimizer.param_groups[0]['lr']:.2e}")

            # Per-scenario breakdown
            sc_improved = False
            for n in scenario_names:
                sc_s = [s for s, sc in zip(success_h[-100:], scenario_h[-100:]) if sc == n]
                if sc_s:
                    rate = sum(sc_s) / len(sc_s)
                    marker = ""
                    if rate > best_per_sc[n] + 0.01:
                        best_per_sc[n] = rate
                        sc_improved    = True
                        marker         = " ↑"
                    print(f"     {n:<15} {rate*100:.1f}%  (best {best_per_sc[n]*100:.1f}%){marker}")

            global_improved = (
                curr_sr > best_success + 0.01
                or (curr_sr >= best_success and curr_steps < best_steps * 0.95)
            )
            improved = global_improved or sc_improved

            if improved:
                best_success = max(best_success, curr_sr)
                best_steps   = min(best_steps, curr_steps)
                best_weights = {
                    'online': copy.deepcopy(agent.online_net.state_dict()),
                    'target': copy.deepcopy(agent.target_net.state_dict()),
                }
                no_improve = 0
                print(f"  ✦ New best: {best_success*100:.1f}% | {best_steps:.1f} avg steps")
            else:
                no_improve += 1
                print(f"  ✗ No improvement ({no_improve}/{patience})")
                if no_improve >= patience:
                    print(f"  Early stopping at episode {ep + 1}")
                    break

            print()

    # Restore best and save
    if best_weights is not None:
        agent.online_net.load_state_dict(best_weights['online'])
        agent.target_net.load_state_dict(best_weights['target'])
        print(f"\n  Restored best: {best_success*100:.1f}% | {best_steps:.1f} avg steps")

    agent.save(save_path)

    total_sr = sum(success_h) / len(success_h) * 100
    print("=" * 65)
    print(f"  Done — {time.time()-t0:.1f}s | Overall success: {total_sr:.1f}%")
    print(f"  Saved → {save_path}")
    print("=" * 65)

    return {
        'rewards_history':   rewards_h,
        'steps_history':     steps_h,
        'success_history':   success_h,
        'stuck_history':     stuck_h,
        'loss_history':      loss_h,
        'best_success_rate': best_success,
        'best_avg_steps':    best_steps,
    }


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    save_path = Path(__file__).resolve().parent.parent / "agents" / "dqn_v2.pt"

    agent = DQNAgentV2(
        learning_rate      = 1e-3,
        discount_factor    = 0.995,
        exploration_rate   = 1.0,
        exploration_min    = 0.02,
        exploration_decay  = 0.9995,
        batch_size         = 128,
        replay_capacity    = 60_000,
        target_update_freq = 300,
        checkpoint_path    = None,
    )

    wind_scenarios = {n: get_wind_scenario(n) for n in SCENARIO_NAMES}

    train_dqn_v2(
        agent,
        wind_scenarios,
        num_episodes    = 3000,
        save_path       = str(save_path),
        biased_explore  = True,
    )