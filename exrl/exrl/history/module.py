import os
import numpy as np
from matplotlib import pyplot as plt
from exrl.utils import save_np_arr

class History:
    def __init__(self, w=20):
        self.w = w
        self.rewards = []
        self.epsilons = []
        self.learning_rates = []

    def add(self, reward, epsilon, lr):
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self.learning_rates.append(lr)

    def moving_average_reward(self):
        x = self.rewards
        w = self.w
        return np.convolve(x, np.ones(w), 'valid') / w

    def plot(self):
        fig, axs = plt.subplots(nrows=2, ncols=1, dpi=300, figsize=(16, 16))
        
        axs[0].set_title("Reward")
        axs[0].plot(self.rewards, label="Rewards")
        axs[0].plot(self.moving_average_reward(), label=f"Average Rewards $ w = {self.w} $")
        
        axs[1].set_title("Epsilon & Learning Rate")
        axs[1].plot(self.epsilons, label="Epsilon")
        axs[1].plot(self.learning_rates, label="Learning Rate")
        
        for ax in axs:
            ax.grid()
            ax.legend()
            ax.set_ylabel("Value")
            ax.set_xlabel("Episode")
            ax.set_xlim(0, len(self.rewards))
        
        return fig

    def log(self) -> dict:
        r = round(self.rewards[-1], 3)
        e = round(self.epsilons[-1], 3)
        mr = np.mean(self.rewards[-self.w:])
        lr = round(self.learning_rates[-1], 3)
        return {
            "R": r,
            "MR": mr,
            "eps": e,
            "lr": lr,
        }

    def save(self, folder: str):
        plots = os.path.join(folder, "history_plots.jpg")
        rewards = os.path.join(folder, "history_rewards.npy")
        epsilons = os.path.join(folder, "history_epsilons.npy")
        learning_rates = os.path.join(folder, "learning_rates.npy")
        
        self.plot().savefig(plots)
        np.save(rewards, np.array(self.rewards))
        np.save(epsilons, np.array(self.epsilons))
        np.save(learning_rates, np.array(self.learning_rates))
        
