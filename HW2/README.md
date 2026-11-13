# Homework 2: A/B Testing: Epsilon-Greedy vs Thompson Sampling

Four ads (bandits) with true mean rewards 1, 2, 3 and 4. I ran 20,000 trials with
Epsilon-Greedy (epsilon = 1/t) and Thompson Sampling (known precision) to see which
one finds the best ad faster.

## How to run

```
pip install -r requirements.txt
python Bandit.py
```

The script saves the CSV files and plots next to `Bandit.py` and logs the results.
The plots also open in a window. Set `SHOW_PLOTS = False` at the top of the file to turn that off.

## Files

- `Bandit.py` - the code
- `rewards.csv` - all rewards (columns: Bandit, Reward, Algorithm)
- `EpsilonGreedy_rewards.csv`, `ThompsonSampling_rewards.csv` - same data per algorithm
- `plot1_EpsilonGreedy.png`, `plot1_ThompsonSampling.png` - learning process (linear and log axis)
- `plot2_comparison.png` - cumulative reward and regret of both algorithms

## How it works

- **Epsilon-Greedy:** with probability 1/t play a random bandit, otherwise play the one with the best average so far.
- **Thompson Sampling:** each bandit has a Normal posterior (weak prior: mean 0, precision 0.01). Every trial I
  sample from each posterior and play the highest sample. With known precision tau, a reward x updates the
  posterior like this: `mu = (tau*x + lambda*mu) / (lambda + tau)` and `lambda = lambda + tau`.
- **Regret:** best true mean (4) minus the true mean of the bandit that was played.

## Results (seed 42)

| | Avg reward | Avg regret | Cumulative reward | Cumulative regret |
|---|---|---|---|---|
| Epsilon-Greedy | 3.695 | 0.309 | 73,907 | 6,181 |
| Thompson Sampling | 3.997 | 0.002 | 79,937 | 37 |

Thompson Sampling is much better. It played the best bandit 19,976 times out of 20,000.

Epsilon-Greedy has flat regret for the first ~6,000 trials. That is not a bug: with 1/t it only explores
about 10 times in total, and here it did not try the best bandit until trial 6,175. Until then it kept
playing bandit 3.

## Bonus: better implementation plan

1. Explore longer in Epsilon-Greedy, for example max(0.01, c/t) or 1/sqrt(t).
2. Run many seeds (about 50) and plot the average regret with a confidence band, because one run can mislead.
3. Learn the precision with a Normal-Gamma prior instead of assuming it is known.
4. Use click-style rewards (0/1) with a Beta-Bernoulli Thompson Sampler, which fits ads better.
5. Compare with a normal A/B test sized by power analysis (with Bonferroni or Holm correction for 4 arms).
6. Move the settings into a config or command-line arguments and add a few small tests.
