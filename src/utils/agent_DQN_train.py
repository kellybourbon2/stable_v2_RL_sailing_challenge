""" Script to train the different agents implemented:
        - DQN agent: with and without sail physics included
        - ...
""" 
import copy
import time
import numpy as np

ACTIONS = np.array([
    ( 0,  1),   # 0 North
    ( 1,  1),   # 1 Northeast
    ( 1,  0),   # 2 East
    ( 1, -1),   # 3 Southeast
    ( 0, -1),   # 4 South
    (-1, -1),   # 5 Southwest
    (-1,  0),   # 6 West
    (-1,  1),   # 7 Northwest
    ( 0,  0),   # 8 Stay
], dtype=np.float32)


def compute_action_efficiencies(wind_vector: np.ndarray) -> np.ndarray:
    wind_norm = np.linalg.norm(wind_vector)
    if wind_norm < 1e-8:
        return np.ones(9, dtype=np.float32) * 0.05
    wind_dir     = wind_vector / wind_norm
    efficiencies = np.zeros(9, dtype=np.float32)
    for i, action in enumerate(ACTIONS):
        action_norm = np.linalg.norm(action)
        if action_norm < 1e-8:
            efficiencies[i] = 0.0
            continue
        action_dir = action / action_norm
        cos_angle  = np.clip(np.dot(action_dir, wind_dir), -1.0, 1.0)
        angle_deg  = np.degrees(np.arccos(cos_angle))
        if angle_deg < 45:
            eff = 0.05
        elif angle_deg < 90:
            eff = 0.5 + 0.5 * (angle_deg - 45) / 45
        elif angle_deg <= 135:
            eff = 1.0
        else:
            eff = 1.0 - 0.5 * (angle_deg - 135) / 45
        efficiencies[i] = eff
    return efficiencies


def augment_obs(observation: np.ndarray) -> np.ndarray:
    local_wind   = observation[4:6]
    efficiencies = compute_action_efficiencies(local_wind)
    return np.concatenate([observation, efficiencies]).astype(np.float32)


def train_dqn_agent(agent, wind_scenarios, num_episodes, save_path, physics_sail=False):
    """Train a DQN on the different wind_scenarios,
       with or without sail physics augmentation, and save to path."""

    max_steps      = 500
    scenario_names = list(wind_scenarios.keys())
    num_scenarios  = len(scenario_names)

    rewards_history  = []
    steps_history    = []
    success_history  = []
    loss_history     = []
    scenario_history = []
    stuck_history    = []     # ← track stuck separately for diagnostics

    # ── Initialise OUTSIDE the loop ───────────────────────────────────────────
    best_success_rate = 0.0
    best_avg_steps    = float('inf')
    best_weights      = None
    no_improve        = 0
    patience          = 5
    best_per_scenario = {name: 0.0 for name in scenario_names}

    # ── Adaptive lr/decay thresholds ──────────────────────────────────────────
    lr_reduced        = False   # flag: only reduce lr once
    decay_slowed      = False   # flag: only slow decay once

    agent.seed(42)
    np.random.seed(42)

    start_time = time.time()
    print("=" * 65)
    print(f"  DQN Training Started")
    print(f"  Episodes      : {num_episodes}")
    print(f"  Scenarios     : {scenario_names}")
    print(f"  Device        : {agent.device}")
    print(f"  Physics sail  : {physics_sail}")
    print("=" * 65)

    for episode in range(num_episodes):
        scenario_name   = scenario_names[episode % num_scenarios]
        scenario_params = wind_scenarios[scenario_name]

        env  = SailingEnv(**scenario_params)
        goal = env.goal_position.copy()

        observation, info = env.reset(seed=episode)
        if physics_sail:
            observation = augment_obs(observation)

        prev_dist      = np.linalg.norm(info['position'] - goal)
        total_reward   = 0
        episode_losses = []

        for step in range(max_steps):
            action = agent.act(observation)
            next_observation, reward, done, truncated, info = env.step(action)

            # Reward shaping -----------------
            curr_dist     = np.linalg.norm(info['position'] - goal)
            shaped_reward = reward + (prev_dist - curr_dist) * 0.01  

            if info.get('is_stuck', False):
                shaped_reward = -50.0
            prev_dist = curr_dist

            loss = agent.learn(
                observation, action, shaped_reward,
                next_observation, float(done or truncated),
                scenario=scenario_name                  # ← balanced buffer
            )
            if loss is not None:
                episode_losses.append(loss)

            if physics_sail:
                observation = augment_obs(next_observation)
            else:
                observation = next_observation
            total_reward += shaped_reward

            if done or truncated:
                break

        env.close()

        # ── Correct success tracking — stuck ≠ success ────────────────────────
        is_stuck     = info.get('is_stuck', False)
        reached_goal = done and not is_stuck

        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        success_history.append(reached_goal)        # ← real success only
        stuck_history.append(is_stuck)
        loss_history.append(np.mean(episode_losses) if episode_losses else 0.0)
        scenario_history.append(scenario_name)

        # ── Adaptive exploration decay ─────────────────────────────────────────
        if agent.buffer.ready(agent.batch_size):
            recent_success = sum(success_history[-100:]) / min(len(success_history), 100)

            if recent_success >= 0.85 and not decay_slowed:
                # Slow down decay significantly when doing well
                agent.exploration_decay = 0.9999   # was 0.9995
                decay_slowed = True
                print(f"  ↓ Exploration decay slowed to {agent.exploration_decay} "
                      f"(success={recent_success*100:.1f}%)", flush=True)

            if recent_success < 1.0:
                agent.decay_exploration()

        # ── Every episode: lightweight progress ───────────────────────────────
        if reached_goal:
            status = "✓"
        elif is_stuck:
            status = "~"    # stuck on island — was wrongly counted as success before
        else:
            status = "✗"    # truncated (number of steps > 500)

        print(f"  [{status}] Ep {episode+1:>4} ({scenario_name}) | "
              f"reward: {total_reward:+6.1f} | steps: {step+1:>3} | "
              f"ε: {agent.exploration_rate:.3f}",
              flush=True)

        # ── Every 100 episodes: rolling summary + best model tracking ─────────
        if (episode + 1) % 100 == 0:
            recent_success_list = success_history[-100:]
            recent_stuck_list   = stuck_history[-100:]
            recent_rewards      = rewards_history[-100:]
            recent_losses       = loss_history[-100:]
            recent_steps        = steps_history[-100:]
            elapsed             = time.time() - start_time
            eps_per_sec         = (episode + 1) / elapsed
            eta_sec             = (num_episodes - episode - 1) / eps_per_sec

            curr_success_rate = sum(recent_success_list) / len(recent_success_list)
            curr_stuck_rate   = sum(recent_stuck_list)   / len(recent_stuck_list)
            curr_avg_steps    = np.mean(recent_steps)

            print()
            print(f"  ── Episode {episode+1}/{num_episodes} ── {elapsed:.0f}s elapsed | ETA: {eta_sec:.0f}s")
            print(f"     Success rate  (last 100): {curr_success_rate*100:.1f}%")
            print(f"     Stuck rate    (last 100): {curr_stuck_rate*100:.1f}%")   # ← new
            print(f"     Avg reward    (last 100): {np.mean(recent_rewards):.2f}  "
                  f"[min {np.min(recent_rewards):.1f} / max {np.max(recent_rewards):.1f}]")
            print(f"     Avg loss      (last 100): {np.mean(recent_losses):.4f}")
            print(f"     Avg steps     (last 100): {curr_avg_steps:.1f}")
            print(f"     Exploration ε           : {agent.exploration_rate:.4f}")
            print(f"     LR                      : {agent.optimizer.param_groups[0]['lr']:.2e}")

            # ── Adaptive learning rate reduction ──────────────────────────────
            if curr_success_rate >= 0.85 and not lr_reduced:
                old_lr = agent.optimizer.param_groups[0]['lr']
                for pg in agent.optimizer.param_groups:
                    pg['lr'] *= 0.2     # 1e-4 → 2e-5
                lr_reduced = True
                print(f"  ↓ LR reduced {old_lr:.2e} → "
                      f"{agent.optimizer.param_groups[0]['lr']:.2e} "
                      f"(success={curr_success_rate*100:.1f}%)")

            # Per-scenario breakdown + individual improvement detection
            scenario_improved = False
            for name in scenario_names:
                sc_successes = [
                    s for s, sc in zip(success_history[-100:], scenario_history[-100:])
                    if sc == name
                ]
                sc_stucks = [
                    s for s, sc in zip(stuck_history[-100:], scenario_history[-100:])
                    if sc == name
                ]
                if sc_successes:
                    sc_rate      = sum(sc_successes) / len(sc_successes)
                    sc_stuck_pct = sum(sc_stucks) / len(sc_stucks) * 100
                    prev_best    = best_per_scenario[name]
                    improved_marker = ""
                    if sc_rate > prev_best + 0.01:
                        best_per_scenario[name] = sc_rate
                        scenario_improved = True
                        improved_marker = " ↑"
                    print(f"     {name:<15} success: {sc_rate*100:.1f}%  "
                          f"stuck: {sc_stuck_pct:.1f}%  "
                          f"(best: {prev_best*100:.1f}%){improved_marker}  "
                          f"({len(sc_successes)} eps)")

            # Global OR per-scenario improvement triggers a save
            global_improved = (
                curr_success_rate > best_success_rate + 0.01
                or (curr_success_rate >= best_success_rate
                    and curr_avg_steps < best_avg_steps * 0.95)
            )
            improved = global_improved or scenario_improved

            if improved:
                if global_improved:
                    best_success_rate = curr_success_rate
                    best_avg_steps    = curr_avg_steps
                best_weights = {
                    'online': copy.deepcopy(agent.online_net.state_dict()),
                    'target': copy.deepcopy(agent.target_net.state_dict()),
                }
                no_improve = 0
                reason = []
                if global_improved:
                    reason.append(f"overall {best_success_rate*100:.1f}%")
                if scenario_improved:
                    gained = [n for n in scenario_names
                              if best_per_scenario[n] > 0]
                    reason.append(f"scenario ↑ {gained}")
                print(f"  ✦ New best saved ({' | '.join(reason)}) | {best_avg_steps:.1f} avg steps")
            else:
                no_improve += 1
                print(f"  ✗ No improvement ({no_improve}/{patience})")
                if no_improve >= patience:
                    print(f"  Early stopping at episode {episode+1}")
                    break
            print()

        # ── Every 500 episodes: milestone banner ──────────────────────────────
        if (episode + 1) % 500 == 0:
            overall_so_far  = sum(success_history) / len(success_history) * 100
            overall_stuck   = sum(stuck_history)   / len(stuck_history)   * 100
            print("  " + "─" * 61)
            print(f"  MILESTONE — {episode+1}/{num_episodes} episodes complete")
            print(f"  Overall success rate so far : {overall_so_far:.1f}%")
            print(f"  Overall stuck rate so far   : {overall_stuck:.1f}%")
            print(f"  Q-network target syncs      : {agent._step_count // agent.target_update_freq}")
            print(f"  Replay buffer size          : {len(agent.buffer)}")
            print("  " + "─" * 61)
            print()

    # ── Restore best weights BEFORE saving ────────────────────────────────────
    if best_weights is not None:
        agent.online_net.load_state_dict(best_weights['online'])
        agent.target_net.load_state_dict(best_weights['target'])
        print(f"\n  Restored best model: {best_success_rate*100:.1f}% | {best_avg_steps:.1f} avg steps")

    training_time        = time.time() - start_time
    overall_success_rate = sum(success_history) / len(success_history) * 100
    overall_stuck_rate   = sum(stuck_history)   / len(stuck_history)   * 100
    per_scenario_success = {
        name: sum(s for s, sc in zip(success_history, scenario_history) if sc == name) /
              max(1, sum(1 for sc in scenario_history if sc == name)) * 100
        for name in scenario_names
    }
    per_scenario_stuck = {
        name: sum(s for s, sc in zip(stuck_history, scenario_history) if sc == name) /
              max(1, sum(1 for sc in scenario_history if sc == name)) * 100
        for name in scenario_names
    }

    agent.save(save_path)

    print("=" * 65)
    print(f"  Training Complete — {training_time:.1f}s total")
    print(f"  Overall success rate : {overall_success_rate:.1f}%")
    print(f"  Overall stuck rate   : {overall_stuck_rate:.1f}%")
    for name in scenario_names:
        print(f"    {name:<18}: success {per_scenario_success[name]:.1f}%  "
              f"stuck {per_scenario_stuck[name]:.1f}%")
    print(f"  Model saved to       : {save_path}")
    print("=" * 65)

    return {
        'rewards_history':        rewards_history,
        'steps_history':          steps_history,
        'success_history':        success_history,
        'stuck_history':          stuck_history,
        'loss_history':           loss_history,
        'scenario_history':       scenario_history,
        'overall_success_rate':   overall_success_rate,
        'overall_stuck_rate':     overall_stuck_rate,
        'per_scenario_success':   per_scenario_success,
        'per_scenario_stuck':     per_scenario_stuck,
        'training_time_sec':      training_time,
        'final_exploration_rate': agent.exploration_rate,
        'best_success_rate':      best_success_rate,
        'best_avg_steps':         best_avg_steps,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))

    from src.wind_scenarios import get_wind_scenario
    from src.env_sailing import SailingEnv
    from src.agents.agent_DQN import DQNAgent

    sample_env  = SailingEnv(**get_wind_scenario('training_1'))
    obs, info   = sample_env.reset(seed=42)
    num_actions = sample_env.action_space.n
    sample_env.close()

    agent = DQNAgent(num_actions=num_actions, checkpoint_path=None)

    wind_scenarios = {
        'training_1': get_wind_scenario('training_1'),
        'training_2': get_wind_scenario('training_2'),
        'training_3': get_wind_scenario('training_3'),
    }

    metrics = train_dqn_agent(
        agent,
        wind_scenarios,
        num_episodes=2000,
        physics_sail=True,
        save_path='dqn_agent_early_stopping_balanced_buffer_physics_sail.pt'
    )