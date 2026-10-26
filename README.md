# Credit Rating Transitions as a Markov Chain

Models a company's credit rating as a Markov chain with states AAA, AA, A, BBB, BB, B, CCC and D (default). It uses the one-year transition matrix from the CreditMetrics Technical Document (J.P. Morgan, 1997), Table 6.2, stored in [data/](data/creditmetrics_1997_transition_matrix.csv). Any matrix with the same layout, such as a recent S&P table, can be dropped in.

The script has three parts:

1. **Powers of P.** Checks that P is a Markov matrix and uses Pⁿ to get n-year default probabilities. Also finds the steady state and the rate of convergence to it from the eigenvalues.
2. **The fundamental matrix N = (I − Q)⁻¹.** Default is an absorbing state, so N gives the expected number of years at each rating before default and the expected time to default. Simulated chains check the result.
3. **Monte Carlo for a bond portfolio.** Simulates 100 bonds over 5 years and checks the simulated default rates against Pⁿ, and the simulated loss distribution, VaR and expected shortfall against their exact values.

## Results

- **Convention:** credit risk uses the row convention x_{n+1} = x_n P, so each row sums to 1. Many linear algebra courses use columns, x_{n+1} = A x_n, which is the transpose.
- **Data quirk:** the published B row sums to 99.99% and the CCC row to 100.01% because of rounding. Each row is rescaled by its sum.
- **n-year default probabilities** are entry (i, D) of Pⁿ. Over 5 years: BBB 2.10%, BB 8.67%, B 24.4%, CCC 54.2%. AAA has a 0% one-year default probability but a positive one from year 2 on, since it can be downgraded first and then default. A single matrix entry can't show this; Pⁿ can.
- **Steady state:** the eigenvector of Pᵀ for λ = 1 puts all mass on D, so every company eventually defaults. The second eigenvalue λ₂ = 0.9882 sets how fast: the survival ratio S(n)/S(n−1) is 0.98 to 0.99 at n = 30 and exactly 0.9882 for every rating by n = 200.
- **Fundamental matrix** N = (I − Q)⁻¹, where Q is the 7 × 7 block of moves between non-default ratings. N[i, j] is the expected number of years spent at rating j before default, and the row sums are the expected years to default: AAA 119, BBB 86, BB 64, B 42, CCC 25. The distributions are very skewed (BBB has a median of 59 years and a standard deviation of 83). 20,000 simulated chains per rating agree, with |z| < 1.6 throughout.
- **Portfolio Monte Carlo:** 100 bonds (5 AAA, 10 AA, 25 A, 30 BBB, 15 BB, 10 B, 5 CCC), $1M face each, 40% recovery, 5 years, 100,000 scenarios in about 2.6 s. Ratings are sampled by inverse CDF on each row of P.
  - simulated default frequencies match Pⁿ for every rating at 1 and 5 years (|z| < 1.8);
  - expected loss is **$4.35M** against an exact $4.36M (z = −0.9). 70% of it comes from the 15 B and CCC bonds;
  - VaR 99% = **$7.8M** and VaR 99.9% = **$9.0M**, both equal to the exact values. The exact loss distribution is a Poisson binomial, computed by convolving each bond's Bernoulli default distribution.
- **Limitation:** the bonds move independently, so the loss distribution is narrow (SD $1.35M). Real defaults cluster in recessions. The usual fix is a one-factor Gaussian copula with correlated rating moves, which would fatten the tail and increase VaR. Monte Carlo is needed for that case, because the exact convolution only works for independent bonds.

![Cumulative default probability](plots/credit_cumulative_default.png)
![Time to default](plots/credit_time_to_default.png)
![Portfolio loss distribution](plots/credit_loss_distribution.png)

## Run

```bash
pip install numpy pandas matplotlib
python credit_rating_markov_chain.py
```

The script prints its tables and saves its plots to `plots/`. Numbers above are from seed 42.
