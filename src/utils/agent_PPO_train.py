# train_ppo_agent.py
"""Script to train PPO agent on sailing environment."""
import copy
import time
import numpy as np


def train_ppo_agent(agent, wind_scenarios, num_episodes, save_path, physics_sail=False):
    """
    Train a PPO agent across multiple wind scenarios.

    Key structural difference from DQN training:
        DQN  : update every step (on random mini-batch from replay buffer)
        PPO  : collect rollout_steps transitions → update → clear buffer → repeat

    The rollout spans multiple episodes and scenarios — PPO doesn't care
    about episode boundaries during collection, only during advantage computation.
    """
    max_steps      = 500
    scenario_names = list(wind_scenarios.keys())
    num_scenarios  = len(scenario_names)

    rewards_history      = []
    steps_history        = []
    success_history      = []
    stuck_history        = []
    scenario_history     = []
    policy_loss_history  = []
    value_loss_history   = []
    entropy_history      = []

    # ── Initialise OUTSIDE the loop ───────────────────────────────────────────
    best_success_rate = 0.0
    best_avg_steps    = float('inf')
    best_weights      = None
    no_improve        = 0
    patience          = 5
    best_per_scenario = {name: 0.0 for name in scenario_names}

    lr_reduced   = False
    decay_slowed = False   # not used in PPO (no ε) but kept for symmetry

    agent.seed(42)
    np.random.seed(42)

    start_time = time.time()
    print("=" * 65)
    print(f"  PPO Training Started")
    print(f"  Episodes      : {num_episodes}")
    print(f"  Scenarios     : {scenario_names}")
    print(f"  Device        : {agent.device}")
    print(f"  Physics sail  : {physics_sail}")
    print(f"  Rollout steps : {agent.rollout_steps}")
    print(f"  PPO epochs    : {agent.n_epochs}")
    print("=" * 65)

    steps_since_update = 0   # global step counter for rollout trigger
    last_update_losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

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

        for step in range(max_steps):

            # ── PPO uses stochastic policy during training ────────────────────
            action, log_prob, value = agent.act_train(observation)
            next_observation, reward, done, truncated, info = env.step(action)

            # Reward shaping
            curr_dist     = np.linalg.norm(info['position'] - goal)
            shaped_reward = reward + (prev_dist - curr_dist) * 0.05
            if info.get('is_stuck', False):
                shaped_reward = -5.0
            prev_dist = curr_dist

            if physics_sail:
                next_obs_store = augment_obs(next_observation)
            else:
                next_obs_store = next_observation

            # Store transition in rollout buffer
            agent.store(
                obs      = observation,
                action   = action,
                log_prob = log_prob,
                reward   = shaped_reward,
                value    = value,
                done     = float(done or truncated),
                scenario = scenario_name,
            )

            observation    = next_obs_store
            total_reward  += reward
            steps_since_update += 1

            # ── PPO update: triggered every rollout_steps ─────────────────────
            # Unlike DQN (updates every step), PPO waits until buffer is full
            if steps_since_update >= agent.rollout_steps:
                last_update_losses = agent.update(observation)
                steps_since_update = 0
                print(f"  [UPDATE #{agent._update_count}] "
                      f"policy_loss: {last_update_losses['policy_loss']:.4f} | "
                      f"value_loss: {last_update_losses['value_loss']:.4f} | "
                      f"entropy: {last_update_losses['entropy']:.4f}",
                      flush=True)

            if done or truncated:
                break

        env.close()

        # ── Correct success tracking ───────────────────────────────────────────
        is_stuck     = info.get('is_stuck', False)
        reached_goal = done and not is_stuck

        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        success_history.append(reached_goal)
        stuck_history.append(is_stuck)
        scenario_history.append(scenario_name)
        policy_loss_history.append(last_update_losses['policy_loss'])
        value_loss_history.append(last_update_losses['value_loss'])
        entropy_history.append(last_update_losses['entropy'])

        # ── Every episode: lightweight progress ───────────────────────────────
        if reached_goal:
            status = "✓"
        elif is_stuck:
            status = "~"
        else:
            status = "✗"

        print(f"  [{status}] Ep {episode+1:>4} ({scenario_name}) | "
              f"reward: {total_reward:+6.1f} | steps: {step+1:>3} | "
              f"updates: {agent._update_count}",
              flush=True)

        # ── Every 100 episodes: rolling summary + best model tracking ─────────
        if (episode + 1) % 100 == 0:
            recent_success_list = success_history[-100:]
            recent_stuck_list   = stuck_history[-100:]
            recent_rewards      = rewards_history[-100:]
            recent_steps        = steps_history[-100:]
            recent_policy_loss  = policy_loss_history[-100:]
            recent_value_loss   = value_loss_history[-100:]
            recent_entropy      = entropy_history[-100:]
            elapsed             = time.time() - start_time
            eps_per_sec         = (episode + 1) / elapsed
            eta_sec             = (num_episodes - episode - 1) / eps_per_sec

            curr_success_rate = sum(recent_success_list) / len(recent_success_list)
            curr_stuck_rate   = sum(recent_stuck_list)   / len(recent_stuck_list)
            curr_avg_steps    = np.mean(recent_steps)

            print()
            print(f"  ── Episode {episode+1}/{num_episodes} ── {elapsed:.0f}s elapsed | ETA: {eta_sec:.0f}s")
            print(f"     Success rate  (last 100): {curr_success_rate*100:.1f}%")
            print(f"     Stuck rate    (last 100): {curr_stuck_rate*100:.1f}%")
            print(f"     Avg reward    (last 100): {np.mean(recent_rewards):.2f}  "
                  f"[min {np.min(recent_rewards):.1f} / max {np.max(recent_rewards):.1f}]")
            print(f"     Policy loss   (last 100): {np.mean(recent_policy_loss):.4f}")
            print(f"     Value loss    (last 100): {np.mean(recent_value_loss):.4f}")
            print(f"     Entropy       (last 100): {np.mean(recent_entropy):.4f}")
            print(f"     Avg steps     (last 100): {curr_avg_steps:.1f}")
            print(f"     PPO updates so far      : {agent._update_count}")
            print(f"     LR                      : {agent.optimizer.param_groups[0]['lr']:.2e}")

            # ── Adaptive LR reduction ─────────────────────────────────────────
            if curr_success_rate >= 0.85 and not lr_reduced:
                old_lr = agent.optimizer.param_groups[0]['lr']
                for pg in agent.optimizer.param_groups:
                    pg['lr'] *= 0.3     # 3e-4 → ~1e-4
                lr_reduced = True
                print(f"  ↓ LR reduced {old_lr:.2e} → "
                      f"{agent.optimizer.param_groups[0]['lr']:.2e} "
                      f"(success={curr_success_rate*100:.1f}%)")

            # Per-scenario breakdown
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

            # Global OR per-scenario improvement
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
                best_weights = copy.deepcopy(agent.network.state_dict())
                no_improve   = 0
                reason = []
                if global_improved:
                    reason.append(f"overall {best_success_rate*100:.1f}%")
                if scenario_improved:
                    gained = [n for n in scenario_names if best_per_scenario[n] > 0]
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
            overall_so_far = sum(success_history) / len(success_history) * 100
            overall_stuck  = sum(stuck_history)   / len(stuck_history)   * 100
            print("  " + "─" * 61)
            print(f"  MILESTONE — {episode+1}/{num_episodes} episodes complete")
            print(f"  Overall success rate so far : {overall_so_far:.1f}%")
            print(f"  Overall stuck rate so far   : {overall_stuck:.1f}%")
            print(f"  PPO updates performed       : {agent._update_count}")
            print("  " + "─" * 61)
            print()

    # ── Restore best weights BEFORE saving ────────────────────────────────────
    if best_weights is not None:
        agent.network.load_state_dict(best_weights)
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
        'rewards_history':       rewards_history,
        'steps_history':         steps_history,
        'success_history':       success_history,
        'stuck_history':         stuck_history,
        'scenario_history':      scenario_history,
        'policy_loss_history':   policy_loss_history,
        'value_loss_history':    value_loss_history,
        'entropy_history':       entropy_history,
        'overall_success_rate':  overall_success_rate,
        'overall_stuck_rate':    overall_stuck_rate,
        'per_scenario_success':  per_scenario_success,
        'per_scenario_stuck':    per_scenario_stuck,
        'training_time_sec':     training_time,
        'best_success_rate':     best_success_rate,
        'best_avg_steps':        best_avg_steps,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))

    from src.wind_scenarios import get_wind_scenario
    from src.env_sailing import SailingEnv
    from src.agents.agent_PPO import PPOAgent

    # augment_obs imported if physics_sail=True
    def augment_obs(obs): return obs   # placeholder — replace with real one if needed

    sample_env  = SailingEnv(**get_wind_scenario('training_1'))
    obs, info   = sample_env.reset(seed=42)
    num_actions = sample_env.action_space.n
    sample_env.close()

    agent = PPOAgent(num_actions=num_actions, checkpoint_path=None)

    wind_scenarios = {
        'training_1': get_wind_scenario('training_1'),
        'training_2': get_wind_scenario('training_2'),
        'training_3': get_wind_scenario('training_3'),
    }

    metrics = train_ppo_agent(
        agent,
        wind_scenarios,
        num_episodes=2000,
        save_path='ppo_agent.pt',
        physics_sail=False,
    )