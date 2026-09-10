import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from simple_rl_implementation.epsilon_greedy_policy import Policy

class Training:
    @staticmethod
    def training(taxi_env):
        q_table = np.zeros(shape=(500, 6))
        # Parameters
        EPSILON = 0.1
        ALPHA = 0.1
        GAMMA = 0.9
        NUM_EPISODES = 5000
        # Store training results
        rewards = []
        episode_lengths = []
        for episode in range(NUM_EPISODES):
            done = False
            total_reward = 0
            episode_length = 0
            state, _ = taxi_env.reset()
            action = Policy.policy(q_table, state, EPSILON)
            while not done:
                next_state, reward, terminated, truncated, _ = taxi_env.step(action)
                done = terminated or truncated
                reward = float(reward)
                next_action = Policy.policy(q_table,next_state,EPSILON)
                # SARSA update
                q_table[state][action] = (q_table[state][action]+ ALPHA * (reward+ GAMMA * q_table[next_state][next_action]- q_table[state][action]) )
                state = next_state
                action = next_action
                total_reward += reward
                episode_length += 1
            # Store results
            rewards.append(total_reward)
            episode_lengths.append(episode_length)
            print("Episode:", episode,"Episode Length:", episode_length,"Total Reward:", total_reward)

        taxi_env.close()
        pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
        print("Training Complete. Q Table Saved")

        episodes = range(1, NUM_EPISODES + 1)
        plt.figure()
        plt.plot(episodes, rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SARSA Training - Reward per Episode")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(episodes, episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("SARSA Training - Episode Length")
        plt.grid(True)
        plt.show()