import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from simple_rl_implementation.epsilon_greedy_policy import Policy

class Evaluator:
    def __init__(self, taxi_env,q_table_path="sarsa_q_table.pkl"):
        self.env=taxi_env
        self.q_table = pkl.load(open(q_table_path, "rb"))
        self.NUM_EPISODES = 5
        self.delay = 500
        # Store results for graphs
        self.rewards = []
        self.episode_lengths = []

    def initialize_frame(self):
        width, height = 1000, 640
        frame = np.ones((height, width, 3), dtype=np.uint8) * 120

        GRASS = (45, 170, 25)

        # Outer grass barriers
        cv2.rectangle(frame, (90, 25), (900, 75), GRASS, -1)
        cv2.rectangle(frame, (90, 565), (900, 615), GRASS, -1)
        cv2.rectangle(frame, (20, 100), (70, 535), GRASS, -1)
        cv2.rectangle(frame, (920, 100), (970, 535), GRASS, -1)

        # Inner grass barriers
        cv2.rectangle(frame, (380, 100), (430, 270), GRASS, -1)
        cv2.rectangle(frame, (220, 370), (270, 535), GRASS, -1)
        cv2.rectangle(frame, (565, 370), (615, 535), GRASS, -1)

        # Passenger/destination areas
        cv2.rectangle(frame, (90, 110), (185, 205), (40, 40, 210), -1)  # R
        cv2.rectangle(frame, (90, 470), (185, 565), (40, 220, 220), -1)  # Y
        cv2.rectangle(frame, (640, 470), (735, 565), (200, 50, 50), -1)  # B
        cv2.rectangle(frame, (820, 110), (900, 205), (30, 200, 30), -1)  # G

        # Labels
        cv2.putText(frame, "R", (125, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, "Y", (125, 525),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(frame, "B", (675, 525),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, "G", (850, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Building
        cv2.rectangle(frame, (825, 55), (900, 115), (180, 180, 120), -1)
        cv2.rectangle(frame, (820, 48), (905, 62), (210, 200, 120), -1)

        for row in range(2):
            for col in range(3):
                x = 830 + col * 22
                y = 70 + row * 20
                cv2.rectangle(frame, (x, y), (x + 12, y + 10),
                              (100, 150, 160), -1)

        return frame

    def put_agent(self, frame, state):
        taxi_row, taxi_col, passenger, destination =self.env.unwrapped.decode(state)
        # Map Taxi-v4 positions to the parking-lot layout
        x_positions = [130, 310, 480, 680, 860]
        y_positions = [150, 245, 335, 425, 520]
        center_x = x_positions[taxi_col]
        center_y = y_positions[taxi_row]
        # Taxi body
        cv2.rectangle(frame,(center_x - 32, center_y - 15),(center_x + 32, center_y + 15),(0, 215, 255),-1)
        # Taxi roof
        cv2.rectangle(frame,(center_x - 20, center_y - 28),(center_x + 20, center_y - 15),(0, 215, 255),-1)
        # Windows
        cv2.rectangle(frame,(center_x - 15, center_y - 25),(center_x - 2, center_y - 16),(180, 220, 230),-1)
        cv2.rectangle(frame,(center_x + 2, center_y - 25),(center_x + 15, center_y - 16),(180, 220, 230),-1)
        # Wheels
        cv2.circle(frame, (center_x - 22, center_y + 15), 7,(20, 20, 20), -1)
        cv2.circle(frame, (center_x + 22, center_y + 15), 7,(20, 20, 20), -1)
        # Taxi label
        cv2.putText(frame,"TAXI",(center_x - 18, center_y + 5),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0, 0, 0),1)
        return frame

    def run_episode(self, episode):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        while not done:
            frame = self.initialize_frame()
            frame = self.put_agent(frame, state)
            cv2.imshow("Taxi-v4", frame)
            key = cv2.waitKey(self.delay)
            if key == 27:
                return False
            # Select action
            EPSILON=0.0
            action = Policy.policy(self.q_table,state,EPSILON)
            # Environment step
            next_state, reward, terminated, truncated, _ =self.env.step(action)
            reward=float(reward)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            episode_length += 1
        # Store results
        self.rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        print("Episode:", episode,"Length:", episode_length,"Reward:", total_reward)
        return True

    def plot_graphs(self):
        episodes = range(1, len(self.rewards) + 1)
        # Reward graph
        plt.figure()
        plt.plot(episodes,self.rewards,marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Evaluation - Reward per Episode")
        plt.grid(True)
        plt.show()
        # Episode length graph
        plt.figure()
        plt.plot(episodes,self.episode_lengths,marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Evaluation- Episode Length")
        plt.grid(True)
        plt.show()

    def visualize(self):
        for episode in range(self.NUM_EPISODES):
            if not self.run_episode(episode):
                break
        self.env.close()
        cv2.destroyAllWindows()
        # Show graphs
        self.plot_graphs()
