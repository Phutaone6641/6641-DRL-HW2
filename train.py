import torch
import numpy as np

from q_learning import QLearning
# from sarsa import SARSA
# from monte_carlo import MonteCarlo
# from double_q_learning import DoubleQLearning


def train(env, agent, num_episodes=5000):

    rewards_history = []

    for episode in range(num_episodes):

        obs, _ = env.reset()

        state = agent.discretize_state(
            obs[0].cpu().numpy()
        )

        done = False
        total_reward = 0

        episode_memory = []

        while not done:

            # choose discrete action
            action_idx = agent.get_discretize_action(state)

            # convert to continuous force
            force = agent.mapping_action(action_idx)

            action = torch.tensor([[force]], device=env.device)

            # step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)

            next_state = agent.discretize_state(
                next_obs[0].cpu().numpy()
            )

            done = terminated or truncated

            reward = reward.item()

            # update algorithm
            if agent.control_type == "MonteCarlo":

                episode_memory.append((state, action_idx, reward))

            elif agent.control_type == "SARSA":

                next_action = agent.get_discretize_action(next_state)

                agent.update(
                    state,
                    action_idx,
                    reward,
                    next_state,
                    next_action
                )

            elif agent.control_type == "QLearning":

                agent.update(
                    state,
                    action_idx,
                    reward,
                    next_state
                )

            elif agent.control_type == "DoubleQLearning":

                agent.update(
                    state,
                    action_idx,
                    reward,
                    next_state
                )

            state = next_state
            total_reward += reward

        # Monte Carlo update after episode
        if agent.control_type == "MonteCarlo":
            agent.update(episode_memory)

        agent.decay_epsilon()

        rewards_history.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode} Reward {total_reward}")

    # save trained model
    agent.save_q_value("cartpole_q_model.pkl")

    # save training data
    np.save("reward_history.npy", rewards_history)

    return rewards_history

def main(env):

    agent = QLearning(

        control_type="QLearning",

        num_actions=7,
        action_range=[-10, 10],

        discretize_state_weight=[5,5,10,10],

        learning_rate=0.1,

        initial_epsilon=1.0,
        epsilon_decay_rate=0.995,
        final_epsilon=0.05,

        discount_factor=0.99
    )

    train(env, agent)


if __name__ == "__main__":
    main()