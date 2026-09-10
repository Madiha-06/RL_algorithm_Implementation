import numpy as np
import matplotlib.pyplot as plt
from rl_with_langgraph.negotiation_game import NegotiationGame
from rl_with_langgraph.customer import Customer


class Tester:
    STRATEGIES = ["patient", "impatient", "tit_for_tat"]

    def __init__(self, agent, episodes_per_strategy=200, window=20):
        self.agent = agent
        self.episodes_per_strategy = episodes_per_strategy
        self.window = window
        self.game = NegotiationGame(agent)
        self.history = {k: [] for k in ["episode", "reward", "agent_offer", "rounds", "outcome"]}

    def test(self):
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        episode = 0
        for strategy in self.STRATEGIES:
            for _ in range(self.episodes_per_strategy):
                result, reward = self.game.run_episode(Customer(strategy))
                episode += 1
                self.history["episode"].append(episode)
                self.history["reward"].append(reward)
                self.history["agent_offer"].append(result["agent_offer"])
                self.history["rounds"].append(result["round_num"])
                self.history["outcome"].append(result["outcome"])
        self.agent.epsilon = original_epsilon

    def _moving_average(self, values):
        w = self.window
        return np.convolve(np.asarray(values, dtype=float), np.ones(w) / w, mode="valid")

    def _plot(self, y, title, ylabel, with_average=False):
        plt.figure(figsize=(12, 5))
        x = self.history["episode"]
        plt.plot(x, y, alpha=0.3, label="Raw" if with_average else None)
        if with_average:
            avg = self._moving_average(y)
            plt.plot(x[self.window - 1:], avg, linewidth=2, label=f"{self.window} Episode Average")
            plt.legend()
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_reward(self):
        self._plot(self.history["reward"], "Test Reward", "Reward", with_average=True)

    def plot_final_offer(self):
        self._plot(self.history["agent_offer"], "Agent Final Offer During Test", "Final Offer", with_average=True)

    def plot_deal_rate(self):
        deal_values = [1 if o == "deal" else 0 for o in self.history["outcome"]]
        deal_rate = self._moving_average(deal_values)
        plt.figure(figsize=(12, 5))
        plt.plot(self.history["episode"][self.window - 1:], deal_rate)
        plt.xlabel("Episode")
        plt.ylabel("Deal Rate")
        plt.title("Deal Rate During Test")
        plt.grid(True)
        plt.show()

    def plot_all(self):
        self.plot_reward()
        self.plot_final_offer()
        self.plot_deal_rate()



