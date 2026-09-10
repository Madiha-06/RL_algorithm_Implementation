import gymnasium as gym
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, taxi_env,q_table_path="sarsa_q_table.pkl"):
        self.env = taxi_env
        self.q_table =pkl.load(open(q_table_path, "rb"))
        self.NUM_EPISODES = 10

    def test_policy(self, state):
        # Greedy action: no exploration
        action = int(np.argmax(self.q_table[state]))
        return action

    def test_episode(self, episode):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            # Select action using trained Q-table
            action = self.test_policy(state)
            # Take action
            next_state, reward, terminated, truncated, _ =self.env.step(action)
            reward=float(reward)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            episode_length += 1
        return total_reward, episode_length

    def test(self):
        rewards = []
        episode_lengths = []
        successful_episodes = 0
        for episode in range(self.NUM_EPISODES):
            reward, length = self.test_episode(episode)
            rewards.append(reward)
            episode_lengths.append(length)
            # Taxi-v4 gives +20 for successful drop-off
            if reward > 0:
                successful_episodes += 1
            print("Episode:", episode, "Reward:", reward,"Length:", length )
        # Calculate results
        average_reward = np.mean(rewards)
        average_length = np.mean(episode_lengths)
        success_rate = (successful_episodes / self.NUM_EPISODES) * 100
        print("Episodes:", self.NUM_EPISODES)
        print("Average Reward:", average_reward)
        print("Average Episode Length:", average_length)
        print("Successful Episodes:", successful_episodes)
        print("Success Rate:", success_rate, "%")
        self.env.close()
        episodes = range(1, self.NUM_EPISODES + 1)
        # Reward Graph
        plt.figure()
        plt.plot(episodes,rewards,marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Testing - Reward per Episode")
        plt.grid(True)
        plt.show()

        # Episode Length Graph
        plt.figure()
        plt.plot(episodes,episode_lengths, marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Testing - Episode Length")
        plt.grid(True)
        plt.show()