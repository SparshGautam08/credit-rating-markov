# Credit rating transitions as a Markov chain.
# The states are the ratings AAA ... CCC plus default (D), and the one-year
# transition matrix P gives the probability of moving from one rating to
# another in a year. Default is absorbing: once a company defaults it stays there.
# Part 1 checks P is a Markov matrix and uses its powers for n-year default
# probabilities. Part 2 uses the fundamental matrix of the absorbing chain for
# the expected time to default. Part 3 simulates a bond portfolio by Monte
# Carlo and checks the simulated defaults and losses against Parts 1 and 2.
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(HERE, "plots")
# One-year matrix from the CreditMetrics Technical Document (J.P. Morgan, 1997),
# Table 6.2, in percent. Any other matrix with the same layout can be dropped in.
DATA_PATH = os.path.join(HERE, "data", "creditmetrics_1997_transition_matrix.csv")

HORIZONS = (1, 2, 3, 5, 10, 20, 30)
CHAINS_PER_RATING = 20_000    # Part 2 check: chains run until they default

# Part 3 portfolio: number of bonds at each rating, each with face value $1M
PORTFOLIO = {"AAA": 5, "AA": 10, "A": 25, "BBB": 30, "BB": 15, "B": 10, "CCC": 5}
FACE, RECOVERY = 1.0, 0.4     # $M; a default loses (1 - RECOVERY) of face
YEARS = 5
SIMS = 100_000


# Reads P in the row convention x_{n+1} = x_n P used in credit risk, so
# P[i, j] is the probability of moving from rating i to rating j and each row
# sums to 1. (Many linear algebra courses use columns; the two are transposes.)
def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    assert list(df.index) == list(df.columns), "rows and columns must list the same ratings"
    return df.to_numpy() / 100, list(df.columns)


# Published tables are rounded to 0.01%, so some rows sum to 99.99 or 100.01.
# Rescaling each row by its sum is the smallest fix that makes P a Markov matrix.
def normalize_rows(P_raw):
    return P_raw / P_raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- Part 1

def part1(P_raw, labels):
    np.set_printoptions(precision=4, suppress=True)
    d = len(labels) - 1                  # index of the default state
    sums = P_raw.sum(axis=1)
    print("Part 1: the one-year transition matrix")
    print(pd.DataFrame(100 * P_raw, index=labels, columns=labels).to_string(float_format="{:.2f}".format))
    print(f"Row sums before rescaling (%): {dict(zip(labels, np.round(100 * sums, 2)))}")
    P = normalize_rows(P_raw)
    print(f"Markov matrix after rescaling: entries >= 0 {bool((P >= 0).all())}, "
          f"rows sum to 1 {np.allclose(P.sum(axis=1), 1)}, D absorbing {P[d, d] == 1.0}\n")

    # n-year default probability from rating i is entry (i, D) of P^n
    powers = {n: np.linalg.matrix_power(P, n) for n in range(1, max(HORIZONS) + 1)}
    pd_table = pd.DataFrame({f"{n}y": 100 * powers[n][:d, d] for n in HORIZONS}, index=labels[:d])
    print("Cumulative default probability (%), entry (i, D) of P^n")
    print(pd_table.to_string(float_format="{:.2f}".format) + "\n")

    # A BBB company's rating distribution after n years is e_BBB P^n
    bbb = labels.index("BBB")
    x0 = np.eye(len(labels))[bbb]
    dist = pd.DataFrame({f"{n}y": 100 * x0 @ powers[n] for n in (1, 5, 10, 30)}, index=labels).T
    print("Rating distribution of a BBB company after n years (%), x_n = x_0 P^n")
    print(dist.to_string(float_format="{:.2f}".format) + "\n")

    # The steady state solves pi P = pi, so pi is an eigenvector of P^T for
    # eigenvalue 1. With D absorbing it is all mass on D: in the long run every
    # company defaults. How fast is set by the next-largest eigenvalue, since
    # the survival probability 1 - PD(n) shrinks roughly like lambda_2^n.
    vals, vecs = np.linalg.eig(P.T)
    order = np.argsort(-np.abs(vals))
    pi = np.real(vecs[:, order[0]])
    pi /= pi.sum()
    lam2 = np.real(vals[order[1]])
    print(f"Eigenvalues of P: {np.real_if_close(vals[order])}")
    print(f"Steady state (eigenvector for lambda = 1): {dict(zip(labels, np.round(pi, 4)))}")
    surv = {n: 1 - np.linalg.matrix_power(P, n)[:d, d] for n in (29, 30, 199, 200)}
    print(f"lambda_2 = {lam2:.4f}; survival ratio S(n)/S(n-1) by rating:")
    print(f"  n = 30:  {np.round(surv[30] / surv[29], 4)}")
    print(f"  n = 200: {np.round(surv[200] / surv[199], 4)}\n")
    return P, powers


# ---------------------------------------------------------------- Part 2

# Writing P with the transient ratings first, P = [[Q, R], [0, 1]], where Q is
# the 7 x 7 block of moves between non-default ratings. The fundamental matrix
# N = (I - Q)^-1 = I + Q + Q^2 + ... has N[i, j] = expected number of years
# spent at rating j before default, starting from rating i. Its row sums are
# the expected years to default, t = N 1, and Var = (2N - I) t - t^2.
def absorbing_chain(P):
    d = len(P) - 1
    Q = P[:d, :d]
    N = np.linalg.inv(np.eye(d) - Q)
    t = N.sum(axis=1)
    var = (2 * N - np.eye(d)) @ t - t**2
    return N, t, var


# Runs n chains from each rating until every one has defaulted and returns
# the default years. All chains step together, one year at a time.
def simulate_default_times(P, n, rng):
    d = len(P) - 1
    cum = np.cumsum(P, axis=1)
    state = np.repeat(np.arange(d), n)
    years = np.zeros(len(state), dtype=int)
    alive = np.ones(len(state), dtype=bool)
    while alive.any():
        idx = np.flatnonzero(alive)
        state[idx] = step(state[idx], cum, rng)
        years[idx] += 1
        alive[idx] = state[idx] != d
    return years.reshape(d, n)


# One year of rating moves: draw u ~ U(0, 1) and move to the first rating j
# whose cumulative row probability exceeds u (inverse-CDF sampling)
def step(state, cum, rng):
    u = rng.random(len(state))
    nxt = np.empty_like(state)
    for s in np.unique(state):
        m = state == s
        nxt[m] = np.searchsorted(cum[s], u[m], side="right")
    return np.minimum(nxt, cum.shape[1] - 1)   # u at the top of a row that sums to 1 - 1e-16


def part2(P, labels, powers, rng):
    d = len(labels) - 1
    N, t, var = absorbing_chain(P)
    ig = [labels.index(r) for r in ("AAA", "AA", "A", "BBB")]
    # Median: first n with P^n[i, D] >= 1/2
    median = [next(n for n in range(1, 10**4)
                   if (powers[n] if n in powers else np.linalg.matrix_power(P, n))[i, d] >= 0.5)
              for i in range(d)]
    print("Part 2: time to default from the fundamental matrix N = (I - Q)^-1")
    print("Expected years spent at each rating before default, N[i, j]")
    print(pd.DataFrame(N, index=labels[:d], columns=labels[:d]).to_string(float_format="{:.1f}".format))

    t0 = time.perf_counter()
    years = simulate_default_times(P, CHAINS_PER_RATING, rng)
    elapsed = time.perf_counter() - t0
    mc_mean = years.mean(axis=1)
    mc_se = years.std(axis=1, ddof=1) / np.sqrt(CHAINS_PER_RATING)
    df = pd.DataFrame({
        "Expected years": t,
        "SD years": np.sqrt(var),
        "Median years": median,
        "Years in IG": N[:, ig].sum(axis=1),
        "MC mean": mc_mean,
        "MC SE": mc_se,
        "z": (mc_mean - t) / mc_se,
        "MC SD": years.std(axis=1, ddof=1),
    }, index=labels[:d])
    print(f"\nTime to default: exact from N against {CHAINS_PER_RATING:,} simulated chains per rating "
          f"({elapsed:.1f} s)")
    print(df.to_string(float_format="{:.2f}".format, formatters={"Median years": "{:d}".format}) + "\n")
    return t, years


# ---------------------------------------------------------------- Part 3

# Simulates the ratings of every bond in every scenario for YEARS years and
# returns the year each bond defaults (0 = no default within the horizon).
# Bonds move independently of each other.
def simulate_portfolio(P, start, sims, years, rng):
    d = len(P) - 1
    cum = np.cumsum(P, axis=1)
    state = np.tile(start, sims)
    default_year = np.zeros(len(state), dtype=np.int8)
    for y in range(1, years + 1):
        alive = np.flatnonzero(state != d)
        state[alive] = step(state[alive], cum, rng)
        default_year[alive[state[alive] == d]] = y
    return default_year.reshape(sims, len(start))


# Exact distribution of the number of defaults: each bond defaults
# independently with its own probability p_i (a Poisson binomial), so the pmf
# is the convolution of the Bernoulli pmfs [1 - p_i, p_i]
def default_count_pmf(p):
    pmf = np.array([1.0])
    for pi in p:
        pmf = np.convolve(pmf, [1 - pi, pi])
    return pmf


# Loss quantile and expected shortfall (mean loss in the worst 1 - level)
def var_es(losses, level):
    v = np.quantile(losses, level, method="inverted_cdf")
    return v, losses[losses >= v].mean()


def exact_var_es(pmf, lgd, level):
    cdf = np.cumsum(pmf)
    k = np.searchsorted(cdf, level)       # smallest k with P(K <= k) >= level
    tail = np.arange(k, len(pmf))
    return k * lgd, lgd * (pmf[tail] * tail).sum() / pmf[tail].sum()


def part3(P, labels, powers, rng):
    d = len(labels) - 1
    start = np.concatenate([[labels.index(r)] * n for r, n in PORTFOLIO.items()])
    lgd = FACE * (1 - RECOVERY)
    print(f"Part 3: Monte Carlo for a {len(start)}-bond portfolio over {YEARS} years")
    print(f"Holdings: {PORTFOLIO}, face ${FACE:.0f}M each, recovery {RECOVERY:.0%}, {SIMS:,} scenarios")

    t0 = time.perf_counter()
    default_year = simulate_portfolio(P, start, SIMS, YEARS, rng)
    elapsed = time.perf_counter() - t0
    print(f"Simulation time: {elapsed:.2f} s ({SIMS * len(start) * YEARS / elapsed / 1e6:.0f}M bond-years per second)\n")

    # Check 1: simulated cumulative default frequency by starting rating against P^n
    rows = []
    for r in PORTFOLIO:
        cols = start == labels.index(r)
        dy = default_year[:, cols]
        for n in (1, YEARS):
            freq = ((dy > 0) & (dy <= n)).mean()
            exact = powers[n][labels.index(r), d]
            se = np.sqrt(exact * (1 - exact) / dy.size)
            rows.append({"Rating": r, "Years": n, "Simulated %": 100 * freq, "Exact %": 100 * exact,
                         "SE %": 100 * se, "z": (freq - exact) / se if se > 0 else np.nan})
    print("Cumulative default frequency: simulated against entry (i, D) of P^n")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format,
                                       formatters={"z": "{:+.2f}".format}) + "\n")

    # Check 2: the loss distribution against its exact Poisson-binomial form
    count = (default_year > 0).sum(axis=1)
    losses = count * lgd
    p = powers[YEARS][start, d]
    pmf = default_count_pmf(p)
    k = np.arange(len(pmf))
    el_exact = lgd * p.sum()
    sd_exact = lgd * np.sqrt((p * (1 - p)).sum())
    se = losses.std(ddof=1) / np.sqrt(SIMS)
    rows = [{"Measure": "Expected loss", "Monte Carlo": losses.mean(), "Exact": el_exact},
            {"Measure": "SD of loss", "Monte Carlo": losses.std(ddof=1), "Exact": sd_exact}]
    for level in (0.95, 0.99, 0.999):
        v, es = var_es(losses, level)
        ve, ese = exact_var_es(pmf, lgd, level)
        rows.append({"Measure": f"VaR {level:.1%}", "Monte Carlo": v, "Exact": ve})
        rows.append({"Measure": f"ES {level:.1%}", "Monte Carlo": es, "Exact": ese})
    print(f"Portfolio loss over {YEARS} years ($M); exact values convolve each bond's default probability")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))
    print(f"Expected loss z = {(losses.mean() - el_exact) / se:+.2f} (MC SE {se:.4f}); "
          f"P(no defaults) simulated {np.mean(count == 0):.4f}, exact {pmf[0]:.4f}")
    contrib = pd.Series(lgd * p, index=[labels[s] for s in start]).groupby(level=0, sort=False).sum()
    print(f"Expected loss by rating ($M): {contrib.round(3).to_dict()}\n")
    return count, pmf, lgd


# ---------------------------------------------------------------- Plots

def plot_results(P, labels, powers, t, years, count, pmf, lgd):
    os.makedirs(PLOT_DIR, exist_ok=True)
    d = len(labels) - 1
    n = np.arange(1, max(HORIZONS) + 1)

    # Cumulative default curves, and how a BBB company's rating spreads out
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i in range(d):
        pd_n = np.array([powers[k][i, d] for k in n])
        axes[0].semilogy(n, np.where(pd_n > 0, pd_n, np.nan), label=labels[i])  # log scale skips the zeros
    axes[0].set_xlabel("Years n")
    axes[0].set_ylabel("P(default within n years)")
    axes[0].set_title(r"Cumulative default probability, entry $(i, D)$ of $P^n$")
    axes[0].legend(fontsize=8, ncol=2)
    bbb = labels.index("BBB")
    dist = np.array([np.eye(len(labels))[bbb]] + [powers[k][bbb] for k in n])
    axes[1].stackplot(np.arange(len(dist)), dist.T, labels=labels)
    axes[1].set_xlim(0, len(dist) - 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Years n")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(r"Rating distribution of a BBB company, $x_0 P^n$")
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "credit_cumulative_default.png"), dpi=150)

    # Time to default: expected years by rating, and the full distribution for BB
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mc = years.mean(axis=1)
    x = np.arange(d)
    axes[0].bar(x, t, color="C0", alpha=0.7, label=r"Exact, $N\mathbf{1}$")
    axes[0].plot(x, mc, "o", color="black", ms=4, label="Simulated chains")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels[:d])
    axes[0].set_ylabel("Expected years to default")
    axes[0].set_title("Expected time to default by starting rating")
    axes[0].legend(fontsize=8)
    bb = labels.index("BB")
    cap = 150
    pmf_bb = np.diff([0.0] + [np.linalg.matrix_power(P, k)[bb, d] for k in range(1, cap + 1)])
    axes[1].hist(years[bb], bins=np.arange(0.5, cap + 1.5), density=True, alpha=0.6, label="Simulated chains")
    axes[1].plot(np.arange(1, cap + 1), pmf_bb, color="black", lw=1,
                 label=r"Exact, $P^n[i,D] - P^{n-1}[i,D]$")
    axes[1].set_xlim(0, cap)
    axes[1].set_xlabel("Year of default")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"Year of default for a BB company (mean {t[bb]:.0f} years)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "credit_time_to_default.png"), dpi=150)

    # Portfolio loss distribution, simulated against exact
    fig, ax = plt.subplots(figsize=(7, 4.5))
    kmax = int(np.searchsorted(np.cumsum(pmf), 1 - 1e-6)) + 1
    k = np.arange(kmax + 1)
    sim = np.bincount(count, minlength=kmax + 1)[:kmax + 1] / len(count)
    ax.bar(k * lgd, sim, width=0.8 * lgd, alpha=0.6, label=f"Monte Carlo ({len(count):,} scenarios)")
    ax.plot(k * lgd, pmf[:kmax + 1], "o", color="black", ms=3, label="Exact (Poisson binomial)")
    for level, ls in ((0.99, "--"), (0.999, ":")):
        v, _ = exact_var_es(pmf, lgd, level)
        ax.axvline(v, color="red", lw=1, ls=ls, label=f"VaR {level:.1%} = ${v:.1f}M")
    ax.set_xlabel(f"Loss over {YEARS} years ($M)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Loss distribution of the {sum(PORTFOLIO.values())}-bond portfolio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "credit_loss_distribution.png"), dpi=150)
    plt.close("all")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    rng = np.random.default_rng(seed=42)
    P_raw, labels = load_matrix(DATA_PATH)
    P, powers = part1(P_raw, labels)
    t, years = part2(P, labels, powers, rng)
    count, pmf, lgd = part3(P, labels, powers, rng)
    plot_results(P, labels, powers, t, years, count, pmf, lgd)
    print(f"Plots saved to {PLOT_DIR}")
