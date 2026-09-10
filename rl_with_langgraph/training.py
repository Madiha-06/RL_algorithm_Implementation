import matplotlib.pyplot as plt
import numpy as np
import random
from rl_with_langgraph.QLearning_agent import QLearningAgent
from rl_with_langgraph.negotiation_game import NegotiationGame
from rl_with_langgraph.customer import Customer

class Trainer:
    STRATEGIES = ["patient", "impatient", "tit_for_tat"]

    def __init__(self, num_episodes=3000, window=100, print_every=100):
        self.num_episodes = num_episodes
        self.window = window
        self.print_every = print_every
        self.agent = QLearningAgent(actions=NegotiationGame.ACTIONS)
        self.game = NegotiationGame(self.agent)
        self.history = {k: [] for k in
                         ["episode", "reward", "agent_offer", "rounds", "epsilon", "outcome", "strategy"]}

    def train(self):
        for episode in range(self.num_episodes):
            strategy = random.choice(self.STRATEGIES)
            result, reward = self.game.run_episode(Customer(strategy))
            self._record(episode, strategy, result, reward)
            self._print_progress(episode)
        print("\nTraining complete!")
        print("Learned", len(self.agent.q_table), "unique states")

    def _record(self, episode, strategy, result, reward):
        h = self.history
        h["episode"].append(episode + 1)
        h["reward"].append(reward)
        h["agent_offer"].append(result["agent_offer"])
        h["rounds"].append(result["round_num"])
        h["epsilon"].append(self.agent.epsilon)
        h["outcome"].append(result["outcome"])
        h["strategy"].append(strategy)

    def _print_progress(self, episode):
        if (episode + 1) % self.print_every == 0:
            avg_reward = float(np.mean(self.history["reward"][-self.print_every:]))
            print(f"Episode {episode + 1:4d} | Average Reward: {avg_reward:6.2f} | Epsilon: {self.agent.epsilon:.3f}")

    def _moving_average(self, values):
        w = self.window
        return np.convolve(np.asarray(values, dtype=float), np.ones(w) / w, mode="valid")

    def _plot(self, y, title, ylabel, with_average=False):
        plt.figure(figsize=(12, 5))
        x = self.history["episode"]
        plt.plot(x, y, alpha=0.3, label="Raw" if with_average else None)
        if with_average:
            avg = self._moving_average(y)
            plt.plot(x[self.window - 1:], avg, linewidth=2, label="100 Episode Average")
            plt.legend()
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_reward(self):
        self._plot(self.history["reward"], "Q-Learning Training Reward", "Reward", with_average=True)

    def plot_epsilon(self):
        self._plot(self.history["epsilon"], "Epsilon Decay During Training", "Epsilon")

    def plot_final_offer(self):
        self._plot(self.history["agent_offer"], "Agent Final Offer During Training", "Final Offer", with_average=True)

    def plot_deal_rate(self):
        deal_values = [1 if o == "deal" else 0 for o in self.history["outcome"]]
        deal_rate = self._moving_average(deal_values)
        plt.figure(figsize=(12, 5))
        plt.plot(self.history["episode"][self.window - 1:], deal_rate)
        plt.xlabel("Episode")
        plt.ylabel("Deal Rate")
        plt.title("Deal Rate During Training")
        plt.grid(True)
        plt.show()

    def plot_all(self):
        self.plot_reward()
        self.plot_epsilon()
        self.plot_final_offer()
        self.plot_deal_rate()