from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv

# Logger setup
logger.add("bandit_experiment.log", level="DEBUG", format="{time} | {level} | {message}")


class Bandit(ABC):
    """
    Abstract base class for bandit algorithms.

    This class defines the structure for bandit algorithms, including methods
    for initializing, pulling arms, updating beliefs, running experiments, and reporting results.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit.

        Args:
            p (float): The probability of the bandit.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Represent the bandit as a string.

        Returns:
            str: A string representation of the bandit.
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Simulate pulling the arm of the bandit.

        Returns:
            float: The reward obtained from the bandit.
        """
        pass

    @abstractmethod
    def update(self, reward):
        """
        Update the parameters of the bandit based on the observed reward.

        Args:
            reward (float): The observed reward.
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Run an experiment with the bandit.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Generate a report for the bandit's performance.
        """
        pass


class Visualization:
    """
    Utility class for visualizing bandit experiment results.

    Provides methods to plot the learning process and compare algorithms.
    """

    def plot1(self, rewards, optimal_reward, num_trials, title="Algorithm Performance"):
        """
        Visualize the average reward (linear and log scales).

        Args:
            rewards (list): A list of rewards obtained during trials.
            optimal_reward (float): The maximum possible reward.
            num_trials (int): The total number of trials.
        """
        cumulative_rewards = np.cumsum(rewards)
        average_rewards = cumulative_rewards / (np.arange(1, num_trials + 1))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Linear Scale
        ax[0].plot(average_rewards, label="Average Reward")
        ax[0].axhline(optimal_reward, color="r", linestyle="--", label="Optimal Reward")
        ax[0].set_title(f"{title} (Linear Scale)")
        ax[0].set_xlabel("Trials")
        ax[0].set_ylabel("Average Reward")
        ax[0].legend()

        # Log Scale
        ax[1].plot(average_rewards, label="Average Reward")
        ax[1].axhline(optimal_reward, color="r", linestyle="--", label="Optimal Reward")
        ax[1].set_title(f"{title} (Log Scale)")
        ax[1].set_xlabel("Trials")
        ax[1].set_ylabel("Average Reward")
        ax[1].set_xscale("log")
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    def plot2(self, eg_rewards, ts_rewards, optimal_reward, num_trials):
        """
        Compare cumulative rewards and regrets for Epsilon-Greedy and Thompson Sampling.

        Args:
            eg_rewards (list): Rewards from the Epsilon-Greedy algorithm.
            ts_rewards (list): Rewards from the Thompson Sampling algorithm.
            optimal_reward (float): The maximum possible reward.
            num_trials (int): The total number of trials.
        """
        eg_cumulative_rewards = np.cumsum(eg_rewards)
        ts_cumulative_rewards = np.cumsum(ts_rewards)

        eg_regrets = optimal_reward * np.arange(1, num_trials + 1) - eg_cumulative_rewards
        ts_regrets = optimal_reward * np.arange(1, num_trials + 1) - ts_cumulative_rewards

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Cumulative Rewards
        ax[0].plot(eg_cumulative_rewards, label="Epsilon-Greedy")
        ax[0].plot(ts_cumulative_rewards, label="Thompson Sampling")
        ax[0].set_title("Cumulative Rewards Comparison")
        ax[0].set_xlabel("Trials")
        ax[0].set_ylabel("Cumulative Rewards")
        ax[0].legend()

        # Cumulative Regrets
        ax[1].plot(eg_regrets, label="Epsilon-Greedy")
        ax[1].plot(ts_regrets, label="Thompson Sampling")
        ax[1].set_title("Cumulative Regrets Comparison")
        ax[1].set_xlabel("Trials")
        ax[1].set_ylabel("Cumulative Regrets")
        ax[1].legend()

        plt.tight_layout()
        plt.show()


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit algorithm.

    Attributes:
        true_prob (float): The true probability of the bandit.
        m_estimate (float): The estimated mean reward of the bandit.
        N (int): The number of times the bandit has been pulled.
    """    
    def __init__(self, p):
        self.true_prob = p
        self.m_estimate = 0  # Estimated mean
        self.N = 0  # Number of pulls

    def __repr__(self):
        return f"EpsilonGreedy Bandit(true_prob={self.true_prob:.2f}, estimated_mean={self.m_estimate:.2f})"

    def pull(self):
        return 1 if np.random.random() < self.true_prob else 0

    def update(self, reward):
        self.N += 1
        self.m_estimate = (1 - 1.0 / self.N) * self.m_estimate + (1.0 / self.N) * reward

    @classmethod
    def experiment(cls, bandit_probs, num_trials, initial_epsilon=0.1, min_epsilon=0.01):
        """
        Run an experiment using the Epsilon-Greedy algorithm.

        Args:
            bandit_probs (list): A list of true probabilities for the bandits.
            num_trials (int): The number of trials to perform.
            initial_epsilon (float): The initial exploration rate.
            min_epsilon (float): The minimum exploration rate.

        Returns:
            tuple: A list of bandits and a list of rewards.
        """    
        bandits = [cls(p) for p in bandit_probs]
        rewards = []

        for t in range(1, num_trials + 1):
            epsilon = max(initial_epsilon / t, min_epsilon)
            if np.random.random() < epsilon:
                chosen_bandit = np.random.randint(len(bandits))
            else:
                chosen_bandit = np.argmax([b.m_estimate for b in bandits])

            reward = bandits[chosen_bandit].pull()
            bandits[chosen_bandit].update(reward)
            rewards.append(reward)

        return bandits, rewards

    @classmethod
    def report(cls, bandit_probs, num_trials):
        """
        Generate a report for the Epsilon-Greedy experiment.

        Args:
            bandit_probs (list): A list of true probabilities for the bandits.
            num_trials (int): The number of trials to perform.

        Returns:
            list: A list of rewards obtained during the experiment.
        """
        bandits, rewards = cls.experiment(bandit_probs, num_trials)
        optimal_reward = max(bandit_probs)
        
        cumulative_reward = np.sum(rewards)
        average_reward = cumulative_reward / num_trials

        cumulative_regret = optimal_reward * num_trials - cumulative_reward
        average_regret = cumulative_regret / num_trials

        with open("epsilon_greedy_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Trial", "Reward", "Algorithm"])
            for i, reward in enumerate(rewards, start=1):
                writer.writerow([i, reward, "Epsilon-Greedy"])

        logger.info(f"Epsilon-Greedy Cumulative Reward: {cumulative_reward}")
        logger.info(f"Epsilon-Greedy Average Reward: {average_reward:.4f}")
        logger.info(f"Epsilon-Greedy Cumulative Regret: {cumulative_regret}")
        logger.info(f"Epsilon-Greedy Average Regret: {average_regret:.4f}")

        Visualization().plot1(rewards, optimal_reward, num_trials, title="Epsilon-Greedy Algorithm Performance")
        
        return rewards


class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit algorithm.

    Attributes:
        true_prob (float): The true probability of the bandit.
        m (float): The posterior mean of the bandit.
        lambda_ (float): The posterior precision of the bandit.
        tau (float): The known precision of the rewards.
        N (int): The number of times the bandit has been pulled.
        sum_x (float): The cumulative sum of rewards received.
    """
    def __init__(self, p):
        self.true_prob = p
        self.m = 0  # Posterior mean
        self.lambda_ = 1  # Posterior precision
        self.tau = 1  # Known precision of the rewards
        self.N = 0  # Number of pulls
        self.sum_x = 0  # Sum of observed rewards

    def __repr__(self):
        return f"ThompsonSampling Bandit(true_prob={self.true_prob:.2f}, posterior_mean={self.m:.2f})"

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_prob

    def sample(self):
        """
        Sample from the posterior distribution of the reward mean.

        Returns:
            float: A sample from the posterior distribution.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.m = (self.tau * self.sum_x) / self.lambda_
        self.N += 1

    @classmethod
    def experiment(cls, bandit_probs, num_trials):
        """
        Run an experiment using the Thompson Sampling algorithm.

        Args:
            bandit_probs (list): A list of true probabilities for the bandits.
            num_trials (int): The number of trials to perform.

        Returns:
            tuple: A list of bandits and a list of rewards.
        """    
        bandits = [cls(p) for p in bandit_probs]
        rewards = []

        for _ in range(num_trials):
            chosen_bandit = np.argmax([b.sample() for b in bandits])
            reward = bandits[chosen_bandit].pull()
            bandits[chosen_bandit].update(reward)
            rewards.append(reward)

        return bandits, rewards

    @classmethod
    def report(cls, bandit_probs, num_trials):
        """
        Generate a report for the Thompson Sampling experiment.

        Args:
            bandit_probs (list): A list of true probabilities for the bandits.
            num_trials (int): The number of trials to perform.

        Returns:
            list: A list of rewards obtained during the experiment.
        """
        bandits, rewards = cls.experiment(bandit_probs, num_trials)
        optimal_reward = max(bandit_probs)
        
        cumulative_reward = np.sum(rewards)
        average_reward = cumulative_reward / num_trials

        cumulative_regret = optimal_reward * num_trials - cumulative_reward
        average_regret = cumulative_regret / num_trials

        with open("thompson_sampling_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Trial", "Reward", "Algorithm"])
            for i, reward in enumerate(rewards, start=1):
                writer.writerow([i, reward, "Thompson Sampling"])

        logger.info(f"Thompson Sampling Cumulative Reward: {cumulative_reward}")
        logger.info(f"Thompson Sampling Average Reward: {average_reward:.4f}")
        logger.info(f"Thompson Sampling Cumulative Regret: {cumulative_regret}")
        logger.info(f"Thompson Sampling Average Regret: {average_regret:.4f}")

        Visualization().plot1(rewards, optimal_reward, num_trials, title="Thompson Sampling Algorithm Performance")
        
        return rewards


def comparison():
    """
    Compare the performance of Epsilon-Greedy and Thompson Sampling algorithms.

    This function runs experiments for both algorithms, compares their results,
    and visualizes the cumulative rewards and regrets.
    """
    bandit_probs = [0.3, 0.5, 0.8, 0.9]
    num_trials = 20000

    logger.info("Starting comparison of Epsilon-Greedy and Thompson Sampling...")

    eg_rewards = EpsilonGreedy.report(bandit_probs, num_trials)
    ts_rewards = ThompsonSampling.report(bandit_probs, num_trials)

    optimal_reward = max(bandit_probs)
    Visualization().plot2(eg_rewards, ts_rewards, optimal_reward, num_trials)


if __name__ == '__main__':
    """
    Main execution block for running bandit experiments.

    This block sets up the logger and runs the comparison function.
    """

    logger.info("Starting the Bandit experiments...")
    comparison()