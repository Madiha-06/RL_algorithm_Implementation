import random
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def get_state(self, round_num, agent_offer, customer_offer):
        gap = agent_offer - customer_offer
        gap_bucket = max(0, min(int(gap // 10), 5))
        return round_num, gap_bucket

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self, min_epsilon=0.05, decay_rate=0.999):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)