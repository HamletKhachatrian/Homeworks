"""
A/B testing with multi-armed bandits: Epsilon-Greedy vs Thompson Sampling.

Scenario: we have four advertisement options (bandits) and we don't know which
one is the best. Their true mean rewards are [1, 2, 3, 4], but the algorithms
only see noisy rewards, so they have to learn which ad is best while playing.
We run 20000 trials for each algorithm, then report rewards/regrets and make
the plots.

Run:
    python Bandit.py
"""
############################### LOGGER
from abc import ABC, abstractmethod
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

# --- experiment settings --------------------------------------------------
BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
TRUE_PRECISION = 1.0 # known precision (1/sigma^2) of the reward noise
SEED = 42 # fixed seed so we get the same numbers every run
SHOW_PLOTS = True # open plots in a window (they are saved either way)
OUTPUT_DIR = Path(__file__).resolve().parent # save files next to this script

# loguru prints DEBUG by default, which is too noisy here, so we only keep INFO+.
# (change level to "DEBUG" if you want to see the progress every 5000 trials)
logger.remove()
logger.add(sys.stderr, level = "INFO")


class Bandit(ABC):
    """Abstract blueprint that every bandit algorithm has to follow."""
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """Set up the bandit(s) with the true reward values `p`."""
        pass

    @abstractmethod
    def __repr__(self):
        """Return the name of the algorithm (used in plots and logs)."""
        pass

    @abstractmethod
    def pull(self):
        """Play one bandit and return the reward we got."""
        pass

    @abstractmethod
    def update(self):
        """Update what we know about a bandit after seeing a reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Run the whole experiment (all trials)."""
        pass

    @abstractmethod
    def report(self):
        """Save the results to csv and log the average reward and regret."""
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """Plots for one or two finished experiments."""

    def plot1(self, algo, filename = None):
        """Visualize the learning process of one algorithm (linear and log x-axis).

        Top row shows the average reward, bottom row the average regret. The log
        axis is useful because most of the learning happens in the first few
        hundred trials, which are squeezed together on the linear axis.

        Args:
            algo (Bandit): an algorithm that already ran experiment().
            filename (str, optional): where to save the figure. Defaults to
                plot1_<algorithm name>.png next to this script.
        """
        trials = np.arange(1, len(algo.rewards) + 1)
        # running average = cumulative sum divided by number of trials so far
        avg_reward = np.cumsum(algo.rewards) / trials
        avg_regret = np.cumsum(algo.regrets) / trials
        best = max(algo.p)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for col, scale in enumerate(["linear", "log"]):
            ax = axes[0, col]
            ax.plot(trials, avg_reward, label="average reward")
            # red line = what we would get if we always played the best ad
            ax.axhline(best, color = "red", linestyle = "--", label = f"optimal ({best})")
            ax.set_xscale(scale)
            ax.set_title(f"{algo}: average reward ({scale} x-axis)")
            ax.set_xlabel("trial")
            ax.legend()

            ax = axes[1, col]
            ax.plot(trials, avg_regret, color = "orange", label = "average regret")
            ax.set_xscale(scale)
            ax.set_title(f"{algo}: average regret ({scale} x-axis)")
            ax.set_xlabel("trial")
            ax.legend()

        fig.tight_layout()
        filename = filename or OUTPUT_DIR / f"plot1_{algo}.png"
        fig.savefig(filename, dpi = 120)
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)
        logger.info(f"Saved learning-process plot to {filename}")

    def plot2(self, eg, ts, filename = None):
        """Compare E-greedy and Thompson Sampling: cumulative reward and regret.

        Args:
            eg (EpsilonGreedy): finished Epsilon-Greedy experiment.
            ts (ThompsonSampling): finished Thompson Sampling experiment.
            filename (str, optional): where to save the figure. Defaults to
                plot2_comparison.png next to this script.
        """
        fig, axes = plt.subplots(1, 2, figsize = (13, 5))
        for algo in (eg, ts):
            trials = np.arange(1, len(algo.rewards) + 1)
            axes[0].plot(trials, np.cumsum(algo.rewards), label=str(algo))
            axes[1].plot(trials, np.cumsum(algo.regrets), label=str(algo))

        axes[0].set_title("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        for ax in axes:
            ax.set_xlabel("trial")
            ax.legend()
            ax.grid(alpha = 0.3)

        fig.tight_layout()
        filename = filename or OUTPUT_DIR / "plot2_comparison.png"
        fig.savefig(filename, dpi=120)
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)
        logger.info(f"Saved comparison plot to {filename}")

#--------------------------------------#


class _BaseExperiment(Bandit):
    """Shared code for both algorithms (pulling, logging results, report).

    The only thing that differs between Epsilon-Greedy and Thompson Sampling is
    *how they pick the next bandit*, so that part (`choose`) is left to the
    child classes and everything else lives here to avoid copy-pasting.
    Not meant to be used directly.
    """

    name = "Base"

    def __init__(self, p, n_trials = NUMBER_OF_TRIALS, precision = TRUE_PRECISION, seed = SEED):
        """
        Args:
            p (list[float]): true mean reward of every bandit (the algorithm
                never gets to see these, it can only learn them by playing).
            n_trials (int): number of trials.
            precision (float): known precision of the reward noise.
            seed (int): random seed so results are reproducible.
        """
        self.p = list(p)
        self.k = len(self.p)
        self.n_trials = n_trials
        self.precision = precision
        self.rng = np.random.default_rng(seed)

        # what the algorithm *thinks* each bandit is worth (starts at zero)
        self.est_mean = np.zeros(self.k)
        self.counts = np.zeros(self.k, dtype = int)

        # logs of the experiment, one entry per trial
        self.arms = []
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        """Return the algorithm name."""
        return self.name

    def pull(self, arm):
        """Play one bandit and get a noisy reward.

        Args:
            arm (int): index of the bandit to play.

        Returns:
            float: a reward drawn from Normal(p[arm], 1/precision).
        """
        # numpy wants the standard deviation, which is 1 / sqrt(precision)
        return self.rng.normal(self.p[arm], 1.0 / np.sqrt(self.precision))

    def update(self, arm, reward):
        """Update the running mean of the chosen arm.

        Args:
            arm (int): index of the bandit we just played.
            reward (float): the reward we observed.
        """
        self.counts[arm] += 1
        # incremental mean: no need to store all past rewards
        self.est_mean[arm] += (reward - self.est_mean[arm]) / self.counts[arm]

    @abstractmethod
    def choose(self, t):
        """Pick which bandit to play at trial t.

        Args:
            t (int): current trial number, starting from 1.

        Returns:
            int: index of the chosen bandit.
        """

    def experiment(self):
        """Run all trials: choose -> pull -> update -> log."""
        logger.info(f"Starting {self.name} for {self.n_trials} trials")
        best = max(self.p)
        for t in range(1, self.n_trials + 1):
            arm = self.choose(t)
            reward = self.pull(arm)
            self.update(arm, reward)

            self.arms.append(arm)
            self.rewards.append(reward)
            # regret = how much we lost by not playing the best bandit
            self.regrets.append(best - self.p[arm])
            if t % 5000 == 0:
                logger.debug(f"{self.name}: trial {t}, estimates {np.round(self.est_mean, 2)}")
        logger.info(f"Finished {self.name}")

    def report(self):
        """Save results to CSV and log average/cumulative reward and regret.

        Returns:
            pandas.DataFrame: columns Bandit, Reward, Algorithm.
        """
        # "Bandit" column holds the true mean of the ad that was played
        df = pd.DataFrame({
            "Bandit": [self.p[a] for a in self.arms],
            "Reward": self.rewards,
            "Algorithm": self.name,
        })
        csv_name = OUTPUT_DIR / f"{self.name}_rewards.csv"
        df.to_csv(csv_name, index = False)
        logger.info(f"Saved {len(df)} rows to {csv_name}")

        logger.info(f"[{self.name}] average reward: {np.mean(self.rewards):.4f}")
        logger.info(f"[{self.name}] average regret: {np.mean(self.regrets):.4f}")
        logger.info(f"[{self.name}] cumulative reward: {np.sum(self.rewards):.2f}")
        logger.info(f"[{self.name}] cumulative regret: {np.sum(self.regrets):.2f}")
        # handy for sanity checking: a good algorithm should mostly play the best bandit
        logger.info(f"[{self.name}] pulls per bandit: "
                    f"{dict(zip(self.p, self.counts.tolist()))}")
        return df

#--------------------------------------#


class EpsilonGreedy(_BaseExperiment):
    """Epsilon-greedy with epsilon decaying as 1/t."""

    name = "EpsilonGreedy"

    def choose(self, t):
        """Explore with probability 1/t, otherwise play the best-looking bandit.

        Args:
            t (int): current trial number, starting from 1.

        Returns:
            int: index of the chosen bandit.
        """
        # at t = 1 epsilon is 1 (pure exploration) and it shrinks very fast after that.
        # Downside: if the early rewards are unlucky we can get stuck on a bad arm
        epsilon = 1.0 / t
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.k))   # explore: random bandit
        return int(np.argmax(self.est_mean))        # exploit: best estimate so far

#--------------------------------------#


class ThompsonSampling(_BaseExperiment):
    """Thompson Sampling for Gaussian rewards with known precision.

    Each bandit's mean gets a Normal posterior N(mu, 1/lambda). Because the
    reward precision tau is known, the update after seeing reward x is:
        mu     <- (tau * x + lambda * mu) / (lambda + tau)
        lambda <- lambda + tau
    """

    name = "ThompsonSampling"

    def __init__(self, p, n_trials = NUMBER_OF_TRIALS, precision = TRUE_PRECISION,
                 seed = SEED, prior_mean = 0.0, prior_precision = 0.01):
        """
        Args:
            p (list[float]): true mean reward of every bandit.
            n_trials (int): number of trials.
            precision (float): known precision of the reward noise (tau).
            seed (int): random seed.
            prior_mean (float): starting belief about every bandit's mean.
            prior_precision (float): how sure we are about that belief at the
                start. A small value means a wide prior, i.e. we are not sure.
        """
        super().__init__(p, n_trials, precision, seed)
        # weak prior -> wide posterior at the start -> plenty of early exploration
        self.post_mean = np.full(self.k, prior_mean, dtype = float)
        self.post_prec = np.full(self.k, prior_precision, dtype = float)

    def choose(self, t):
        """Sample from each posterior and play the bandit with the highest sample.

        Args:
            t (int): current trial number (not needed here, kept so the
                signature matches EpsilonGreedy).

        Returns:
            int: index of the chosen bandit.
        """
        # bandits we are unsure about have wide posteriors, so they sometimes
        # win the draw. That's how exploration happens "for free"
        samples = self.rng.normal(self.post_mean, 1.0 / np.sqrt(self.post_prec))
        return int(np.argmax(samples))

    def update(self, arm, reward):
        """Update the posterior of the played bandit (known-precision Normal model).

        Args:
            arm (int): index of the bandit we just played.
            reward (float): the reward we observed.
        """
        super().update(arm, reward)   # also keep the plain running mean for reporting
        lam, mu, tau = self.post_prec[arm], self.post_mean[arm], self.precision
        # new mean is a precision-weighted average of the old belief and the new reward
        self.post_mean[arm] = (tau * reward + lam * mu) / (lam + tau)
        self.post_prec[arm] = lam + tau


def comparison(eg, ts):
    """Compare both algorithms visually and in a short table.

    Args:
        eg (EpsilonGreedy): finished experiment.
        ts (ThompsonSampling): finished experiment.
    """
    viz = Visualization()
    viz.plot2(eg, ts)

    # one combined CSV, as required: Bandit, Reward, Algorithm
    combined = pd.concat([eg.report(), ts.report()], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "rewards.csv", index = False)
    logger.info("Saved combined results to rewards.csv")

    summary = pd.DataFrame({
        "avg reward": [np.mean(a.rewards) for a in (eg, ts)],
        "avg regret": [np.mean(a.regrets) for a in (eg, ts)],
        "cum reward": [np.sum(a.rewards) for a in (eg, ts)],
        "cum regret": [np.sum(a.regrets) for a in (eg, ts)],
    }, index = [str(eg), str(ts)]).round(3)
    logger.info("\n" + summary.to_string())

    # lower cumulative regret = lost less reward = better algorithm
    better = min((eg, ts), key = lambda a: np.sum(a.regrets))
    logger.info(f"Lower cumulative regret: {better}")


if __name__ == '__main__':
    eg = EpsilonGreedy(BANDIT_REWARD)
    ts = ThompsonSampling(BANDIT_REWARD)

    eg.experiment()
    ts.experiment()

    viz = Visualization()
    viz.plot1(eg)
    viz.plot1(ts)
    comparison(eg, ts)