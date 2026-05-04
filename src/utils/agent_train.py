""" Script to train the different agents implemented:
        - DQN agent
        - ...
    
""" 
import time
import numpy as np

def train_dqn_agent(agent, wind_scenarios, num_episodes, save_path):
    
    max_steps      = 500
    scenario_names = list(wind_scenarios.keys())
    num_scenarios  = len(scenario_names)

    rewards_history  = []
    steps_history    = []
    success_history  = []
    loss_history     = []
    scenario_history = []

    agent.seed(42)
    np.random.seed(42)

    start_time = time.time()
    print(f"Starting DQN training: {num_episodes} episodes | device: {agent.device}")

    for episode in range(num_episodes):
        scenario_name   = scenario_names[episode % num_scenarios]
        scenario_params = wind_scenarios[scenario_name]

        env  = SailingEnv(**scenario_params)
        goal = env.goal_position.copy()

        observation, info = env.reset(seed=episode)
        prev_dist    = np.linalg.norm(info['position'] - goal)
        total_reward = 0
        episode_losses = []

        for step in range(max_steps):
            action = agent.act(observation)
            next_observation, reward, done, truncated, info = env.step(action)

            # Reward shaping
            curr_dist     = np.linalg.norm(info['position'] - goal)
            shaped_reward = reward + (prev_dist - curr_dist) * 0.1
            if info.get('is_stuck', False):
                shaped_reward = -10.0
            prev_dist = curr_dist

            loss = agent.learn(
                observation, action, shaped_reward,
                next_observation, float(done or truncated)
            )
            if loss is not None:
                episode_losses.append(loss)

            observation   = next_observation
            total_reward += reward

            if done or truncated:
                break

        agent.decay_exploration()

        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        success_history.append(done)
        loss_history.append(np.mean(episode_losses) if episode_losses else 0.0)
        scenario_history.append(scenario_name)

        if (episode + 1) % 500 == 0:
            recent = success_history[-100:]
            print(
                f"Ep {episode+1}/{num_episodes} | "
                f"Success: {sum(recent)/len(recent)*100:.1f}% | "
                f"Avg reward: {np.mean(rewards_history[-100:]):.2f} | "
                f"Loss: {np.mean(loss_history[-100:]):.4f} | "
                f"ε: {agent.exploration_rate:.3f}"
            )

    training_time = time.time() - start_time
    agent.save(save_path)
    print(f"\nModel saved to '{save_path}'")

    overall_success_rate = sum(success_history) / len(success_history) * 100
    per_scenario_success = {
        name: sum(s for s, sc in zip(success_history, scenario_history) if sc == name) /
              max(1, sum(1 for sc in scenario_history if sc == name)) * 100
        for name in scenario_names
    }

    print(f"Training completed in {training_time:.1f}s | Overall success: {overall_success_rate:.1f}%")

    return {
        'rewards_history':        rewards_history,
        'steps_history':          steps_history,
        'success_history':        success_history,
        'loss_history':           loss_history,
        'scenario_history':       scenario_history,
        'overall_success_rate':   overall_success_rate,
        'per_scenario_success':   per_scenario_success,
        'training_time_sec':      training_time,
        'final_exploration_rate': agent.exploration_rate,
    }

if __name__=="__main__":
    import torch 

    #Add src to root
    import sys
    from pathlib import Path
    root= Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))

    #Import utils
    from src.wind_scenarios import get_wind_scenario
    from src.env_sailing import SailingEnv
    from src.agents.agent_DQN import DQNAgent #Change here with agent

    # Infer obs_dim from a sample env reset
    sample_env = SailingEnv(**get_wind_scenario('training_1'))
    obs, info = sample_env.reset(seed=42) #SEED for reproductibility
    obs_dim     = len(obs)
    num_actions = sample_env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dims=(128, 128),
        learning_rate=1e-3,
        discount_factor=0.995,
        exploration_rate=1.0,
    )

    wind_scenarios = {
        'training_1': get_wind_scenario('training_1'),
        'training_2': get_wind_scenario('training_2'),
        'training_3': get_wind_scenario('training_3'),
    }

    metrics = train_dqn_agent(agent, wind_scenarios, num_episodes=2000, save_path='dqn_agent.pt')