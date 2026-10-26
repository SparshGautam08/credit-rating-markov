# Credit Rating Transitions as a Markov Chain

Models a company's credit rating as a Markov chain with states AAA, AA, A, BBB, BB, B, CCC and D (default). It uses the one-year transition matrix from the CreditMetrics Technical Document (J.P. Morgan, 1997), Table 6.2, stored in [data/](data/creditmetrics_1997_transition_matrix.csv). Any matrix with the same layout, such as a recent S&P table, can be dropped in.

The script has four parts:

1. **Powers of P.** Checks that P is a Markov matrix and uses Pⁿ to get n-year default probabilities. Also finds the steady state and the rate of convergence to it from the eigenvalues.
2. **The fundamental matrix N = (I − Q)⁻¹.** Default is an absorbing state, so N gives the expected number of years at each rating before default and the expected time to default. Simulated chains check the result.
3. **Monte Carlo for a bond portfolio.** Simulates 100 bonds over 5 years and checks the simulated default rates against Pⁿ, and the simulated loss distribution, VaR and expected shortfall against their exact values.
4. **Correlated defaults.** Links the bonds through a one-factor Gaussian copula, so bad years hit every bond at once, and compares the loss tail and VaR with the independent case.

## Results

- **Convention:** credit risk uses the row convention x_{n+1} = x_n P, so each row sums to 1. Many linear algebra courses use columns, x_{n+1} = A x_n, which is the transpose.
- **Data quirk:** the published B row sums to 99.99% and the CCC row to 100.01% because of rounding. Each row is rescaled by its sum.
- **n-year default probabilities** are entry (i, D) of Pⁿ. Over 5 years: BBB 2.10%, BB 8.67%, B 24.4%, CCC 54.2%. AAA has a 0% one-year default probability but a positive one from year 2 on, since it can be downgraded first and then default. A single matrix entry can't show this; Pⁿ can.
- **Steady state:** the eigenvector of Pᵀ for λ = 1 puts all mass on D, so every company eventually defaults. The second eigenvalue λ₂ = 0.9882 sets how fast: the survival ratio S(n)/S(n−1) is 0.98 to 0.99 at n = 30 and exactly 0.9882 for every rating by n = 200.
- **Fundamental matrix** N = (I − Q)⁻¹, where Q is the 7 × 7 block of moves between non-default ratings. N[i, j] is the expected number of years spent at rating j before default, and the row sums are the expected years to default: AAA 119, BBB 86, BB 64, B 42, CCC 25. The distributions are very skewed (BBB has a median of 59 years and a standard deviation of 83). 20,000 simulated chains per rating agree, with |z| < 1.6 throughout.
- **Portfolio Monte Carlo:** 100 bonds (5 AAA, 10 AA, 25 A, 30 BBB, 15 BB, 10 B, 5 CCC), $1M face each, 40% recovery, 5 years, 100,000 scenarios in about 2.8 s. Ratings are sampled by inverse CDF on each row of P.
  - simulated default frequencies match Pⁿ for every rating at 1 and 5 years, within about 2 standard errors;
  - expected loss is **$4.356M** against an exact $4.357M (z = −0.2). 70% of it comes from the 15 B and CCC bonds;
  - VaR 99% = **$7.8M** and VaR 99.9% = **$9.0M**, both equal to the exact values. The exact loss distribution is a Poisson binomial, computed by convolving each bond's Bernoulli default distribution.
  - expected shortfall (ES) is the mean loss over the worst 1% or 0.1% of outcomes. Losses come in $0.6M steps, so ES counts only the part of the step at VaR that lies in the tail.
- **Correlated defaults (one-factor Gaussian copula):** each year, bond i's credit quality is X_i = √ρ Z + √(1 − ρ) ε_i. Z is the economy, shared by all bonds in a scenario, and ε_i is the bond's own news. Each X_i is still N(0, 1), so every bond still follows P exactly; only the dependence changes. ρ is the asset correlation, and the Basel II formula for corporate loans uses 0.12 to 0.24.

  | ρ | Expected loss | SD | VaR 99% | VaR 99.9% | ES 99.9% | P(loss > $10M) |
  |---|---|---|---|---|---|---|
  | 0 | 4.35 | 1.35 | 7.8 | 9.0 | 9.3 | 0.01% |
  | 0.1 | 4.35 | 1.97 | 9.6 | 12.0 | 13.2 | 1.0% |
  | 0.2 | 4.36 | 2.52 | 12.0 | 16.2 | 18.4 | 3.2% |
  | 0.3 | 4.37 | 3.03 | 14.4 | 19.8 | 22.6 | 5.5% |

  - **expected loss doesn't change** with ρ (exact $4.357M, all |z| < 1.4), because correlation doesn't change any single bond's chance of default;
  - **the tail does change:** at ρ = 0.2, VaR 99.9% rises from $9.0M to $16.2M (1.8×) and a loss over $10M goes from about 1 in 10,000 to about 1 in 30. Correlation also makes a loss-free 5 years more likely (0.9% against 0.01%), so the distribution spreads out both ways;
  - **check:** over one year, defaults given Z are independent, so the exact loss distribution is a Poisson binomial averaged over Z by Gauss-Hermite quadrature (it agrees with adaptive integration to about 10 decimal places). Every exact VaR falls inside the simulated VaR's 95% order-statistic CI. With 1M scenarios the simulated ES 99% is within 0.01 of exact. Over 5 years ratings also migrate and defaults depend on the whole path of Z, so there's no such formula, and Monte Carlo is the method.

![Cumulative default probability](plots/credit_cumulative_default.png)
![Time to default](plots/credit_time_to_default.png)
![Portfolio loss distribution](plots/credit_loss_distribution.png)
![Correlated losses](plots/credit_correlated_losses.png)

## Run

```bash
pip install numpy pandas scipy matplotlib
python credit_rating_markov_chain.py
```

The script prints its tables and saves its plots to `plots/`. Numbers above are from seed 42.
