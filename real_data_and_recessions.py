# Real-data transition matrices and a recession regime.
# Part 1 turns Fitch's cohort counts (2013-2024, from estimate_fitch_matrices.py)
# into transition matrices: one pooled over all years, one per year, and one
# each for recession and expansion years. Part 2 makes the economy its own
# two-state Markov chain (expansion <-> recession) and simulates the 100-bond
# portfolio with each year's rating moves drawn from that year's matrix.
# The loss distribution is checked against an exact mixture over regime paths.
import os
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from credit_rating_markov_chain import (DATA_PATH, PLOT_DIR, PORTFOLIO, FACE, RECOVERY, YEARS, SIMS,
                                        load_matrix, normalize_rows, step, default_count_pmf,
                                        var_es, exact_var_es, quantile_ci)

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTS_PATH = os.path.join(HERE, "data", "fitch_transition_counts.csv")
LABELS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
SPEC = ["BB", "B", "CCC"]            # speculative grade ("high yield")

# NBER dates the 2020 US recession February to April 2020; it is the only
# recession in the data. The 2013 to 2024 cohorts cover no other.
RECESSION_YEARS = {2020}
# A regime row with fewer issuer-years than this is too thin to estimate on
# its own and falls back to the pooled row (in practice the AAA row)
MIN_OBS = 30

# Calendar years containing at least one month of NBER-dated US recession,
# 1946 to 2024. This is the same definition that makes 2020 the recession
# cohort above. The regime chain is estimated from this sequence.
NBER_RECESSION_YEARS = {1948, 1949, 1953, 1954, 1957, 1958, 1960, 1961, 1969, 1970, 1973, 1974, 1975,
                        1980, 1981, 1982, 1990, 1991, 2001, 2007, 2008, 2009, 2020}
NBER_SPAN = range(1946, 2025)
REGIME_NAMES = ["Expansion", "Recession"]


# ---------------------------------------------------------------- Part 1

def load_counts(path):
    df = pd.read_csv(path)
    return {y: g.set_index("From")[LABELS].to_numpy() for y, g in df.groupby("Year")}, df


# Maximum-likelihood estimate of a Markov matrix from transition counts:
# P[i, j] = n_ij / n_i. Rows with too few observations use `fallback`. The
# default row is added as absorbing.
def estimate(counts, fallback=None, min_obs=1):
    n = counts.sum(axis=1)
    P = np.zeros((len(LABELS), len(LABELS)))
    P[-1, -1] = 1.0
    for i in range(len(n)):
        P[i] = counts[i] / n[i] if n[i] >= min_obs else fallback[i]
    return P, n


# Wilson 95% interval for a proportion; unlike p +/- 1.96 SE it stays inside
# [0, 1] and gives a sensible upper bound when there are zero defaults
def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def part1():
    by_year, df = load_counts(COUNTS_PATH)
    years = sorted(by_year)
    pooled_counts = sum(by_year.values())
    P_pool, n_pool = estimate(pooled_counts)
    rec_counts = sum(by_year[y] for y in years if y in RECESSION_YEARS)
    exp_counts = sum(by_year[y] for y in years if y not in RECESSION_YEARS)
    P_rec, n_rec = estimate(rec_counts, P_pool, MIN_OBS)
    P_exp, n_exp = estimate(exp_counts, P_pool, MIN_OBS)
    P_cm = normalize_rows(load_matrix(DATA_PATH)[0])

    print(f"Part 1: Fitch corporate transition matrices, cohorts {years[0]} to {years[-1]}")
    print(f"{int(pooled_counts.sum()):,} issuer-years, {int(pooled_counts[:, -1].sum())} defaults; "
          f"{int(df.Withdrawn.sum()):,} withdrawals left out")
    print("Pooled one-year matrix (%), all years")
    print(pd.DataFrame(100 * P_pool[:-1], index=LABELS[:-1], columns=LABELS)
          .assign(**{"Issuer-years": n_pool.astype(int)}).to_string(float_format="{:.2f}".format) + "\n")

    # Default rate by year: the economy shows up as a common shock to every rating
    rows = []
    for y in years:
        c = by_year[y]
        spec = [LABELS.index(r) for r in SPEC]
        rows.append({"Year": y, "Issuers": int(c.sum()), "Defaults": int(c[:, -1].sum()),
                     "All %": 100 * c[:, -1].sum() / c.sum(),
                     "Spec grade %": 100 * c[spec, -1].sum() / c[spec].sum(),
                     "Downgrades %": 100 * np.triu(c[:, :-1], 1).sum() / c.sum(),
                     "Upgrades %": 100 * np.tril(c[:, :-1], -1).sum() / c.sum(),
                     "Regime": "Recession" if y in RECESSION_YEARS else ""})
    yearly = pd.DataFrame(rows)
    print("Yearly default, downgrade and upgrade rates (moves within the year, defaults excluded from downgrades)")
    print(yearly.to_string(index=False, float_format="{:.2f}".format) + "\n")

    # One-year default rate by rating: recession against expansion, with Wilson intervals
    rows = []
    for i, r in enumerate(LABELS[:-1]):
        row = {"Rating": r}
        for name, c in (("Expansion", exp_counts), ("Recession", rec_counts)):
            k, n = c[i, -1], c[i].sum()
            lo, hi = wilson(k, n) if n > 0 else (np.nan, np.nan)
            row[f"{name} n"] = int(n)
            row[f"{name} PD %"] = 100 * k / n if n else np.nan
            row[f"{name} 95% CI"] = f"{100 * lo:.2f} to {100 * hi:.2f}" if n else ""
        row["CreditMetrics PD %"] = 100 * P_cm[i, -1]
        rows.append(row)
    print("One-year default rate by rating (%): recession (2020) against expansion years")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.2f}".format) + "\n")
    thin = [LABELS[i] for i in range(len(n_rec)) if n_rec[i] < MIN_OBS]
    print(f"Recession rows with fewer than {MIN_OBS} issuer-years use the pooled row: {thin}")
    print("Recession matrix (%)")
    print(pd.DataFrame(100 * P_rec[:-1], index=LABELS[:-1], columns=LABELS).to_string(float_format="{:.2f}".format))
    print("Expansion matrix (%)")
    print(pd.DataFrame(100 * P_exp[:-1], index=LABELS[:-1], columns=LABELS).to_string(float_format="{:.2f}".format) + "\n")

    # Multi-year default probabilities: pooled Fitch against CreditMetrics,
    # and five years of expansion against five of recession
    rows = []
    for name, P in (("Fitch pooled", P_pool), ("Fitch expansion", P_exp), ("Fitch recession", P_rec),
                    ("CreditMetrics 1997", P_cm)):
        P5 = np.linalg.matrix_power(P, YEARS)
        rows.append({"Matrix": name, **{r: 100 * P5[LABELS.index(r), -1] for r in ("A", "BBB", "BB", "B", "CCC")}})
    print(f"{YEARS}-year default probability (%), entry (i, D) of P^{YEARS}")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.2f}".format) + "\n")
    return P_pool, P_exp, P_rec, P_cm, yearly


# ---------------------------------------------------------------- Part 2

# The economy as a two-state Markov chain, estimated the same way as the
# rating matrices: count year-to-year moves between expansion (0) and
# recession (1) and divide by the row totals
def estimate_regime_chain():
    seq = [int(y in NBER_RECESSION_YEARS) for y in NBER_SPAN]
    counts = np.zeros((2, 2))
    for a, b in zip(seq, seq[1:]):
        counts[a, b] += 1
    return counts / counts.sum(axis=1, keepdims=True), counts


# Steady state of the regime chain: the eigenvector of R^T for eigenvalue 1
def regime_steady_state(R):
    vals, vecs = np.linalg.eig(R.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


# Simulates the regime path and every bond's rating, one year at a time.
# `first` is the distribution of the first year's regime. Given the regime
# path the bonds move independently, but they all share that path, so a
# recession raises every bond's default probability in the same scenario.
def simulate_regime_portfolio(Ps, R, first, start, sims, years, rng):
    d = len(LABELS) - 1
    cums = [np.cumsum(P, axis=1) for P in Ps]
    regime = (rng.random(sims) < first[1]).astype(int)
    regimes = np.empty((sims, years), dtype=int)
    state = np.tile(start, sims)
    default_year = np.zeros(len(state), dtype=np.int8)
    for y in range(1, years + 1):
        if y > 1:
            regime = (rng.random(sims) < R[regime, 1]).astype(int)
        regimes[:, y - 1] = regime
        bond_regime = np.repeat(regime, len(start))
        u = rng.random(len(state))
        for g in (0, 1):
            alive = np.flatnonzero((state != d) & (bond_regime == g))
            state[alive] = step(state[alive], cums[g], u[alive])
            default_year[alive[state[alive] == d]] = y
    return default_year.reshape(sims, len(start)), regimes


# Exact loss distribution: there are only 2^YEARS regime paths. Along a path
# the n-year transition matrix is the product P_{r1} P_{r2} ... P_{rn}, the
# bonds are independent, and the number of defaults is Poisson binomial.
# Weighting each path's pmf by its probability gives the exact mixture.
def exact_regime_pmf(Ps, R, first, start, years):
    pmf = np.zeros(len(start) + 1)
    for path in itertools.product((0, 1), repeat=years):
        prob = first[path[0]] * np.prod([R[a, b] for a, b in zip(path, path[1:])])
        M = np.linalg.multi_dot([Ps[g] for g in path]) if years > 1 else Ps[path[0]]
        pmf += prob * default_count_pmf(M[start, -1])
    return pmf


def part2(P_pool, P_exp, P_rec, rng):
    start = np.concatenate([[LABELS.index(r)] * n for r, n in PORTFOLIO.items()])
    lgd = FACE * (1 - RECOVERY)
    Ps = [P_exp, P_rec]
    R, counts = estimate_regime_chain()
    pi = regime_steady_state(R)
    lam2 = np.trace(R) - 1                # the other eigenvalue of a 2 x 2 Markov matrix
    print("Part 2: regime-switching Monte Carlo")
    print(f"Regime chain estimated from NBER recession years {NBER_SPAN.start}-{NBER_SPAN.stop - 1} "
          f"(rows = from {REGIME_NAMES})")
    print(f"Transition counts:\n{counts.astype(int)}\nRegime matrix:\n{np.round(R, 4)}")
    print(f"Steady state: expansion {pi[0]:.3f}, recession {pi[1]:.3f}; second eigenvalue {lam2:.3f}")
    print(f"Mean length: expansion {1 / R[0, 1]:.1f} years, recession {1 / R[1, 0]:.1f} years. "
          f"The Fitch cohorts have 1 recession year in 12 ({1 / 12:.3f})")
    print(f"Portfolio: {PORTFOLIO}, face ${FACE:.0f}M each, recovery {RECOVERY:.0%}, "
          f"{YEARS} years, {SIMS:,} scenarios each\n")

    cases = [("Pooled matrix, no regimes", [P_pool, P_pool], np.array([1.0, 0.0])),
             ("Regimes, long-run start", Ps, pi),
             ("Regimes, starting in expansion", Ps, np.array([1.0, 0.0])),
             ("Regimes, starting in recession", Ps, np.array([0.0, 1.0]))]
    rows, check, losses = [], [], {}
    for name, mats, first in cases:
        t0 = time.perf_counter()
        dy, regimes = simulate_regime_portfolio(mats, R, first, start, SIMS, YEARS, rng)
        elapsed = time.perf_counter() - t0
        loss = losses[name] = (dy > 0).sum(axis=1) * lgd
        pmf = exact_regime_pmf(mats, R, first, start, YEARS)
        el_exact = lgd * (np.arange(len(pmf)) * pmf).sum()
        se = loss.std(ddof=1) / np.sqrt(SIMS)
        v99, _ = var_es(loss, 0.99)
        v999, es999 = var_es(loss, 0.999)
        rows.append({"Case": name, "Expected loss": loss.mean(), "EL z": (loss.mean() - el_exact) / se,
                     "SD": loss.std(ddof=1), "VaR 99%": v99, "VaR 99.9%": v999, "ES 99.9%": es999,
                     "P(loss > 5)": np.mean(loss > 5),
                     "Recession years": regimes.sum(axis=1).mean() if mats[0] is not mats[1] else np.nan,
                     "Time (s)": elapsed})
        for level in (0.99, 0.999):
            v, es = var_es(loss, level)
            ve, ese = exact_var_es(pmf, lgd, level)
            lo, hi = quantile_ci(loss, level)
            check.append({"Case": name, "Level": f"{level:.1%}", "MC VaR": v, "MC VaR CI": f"{lo:.1f} to {hi:.1f}",
                          "Exact VaR": ve, "MC ES": es, "Exact ES": ese})
    df = pd.DataFrame(rows)
    print(f"Loss over {YEARS} years ($M)")
    print(df.to_string(index=False, float_format="{:.3f}".format,
                       formatters={"EL z": "{:+.2f}".format, "P(loss > 5)": "{:.4f}".format,
                                   "Recession years": "{:.2f}".format, "Time (s)": "{:.1f}".format},
                       na_rep="-") + "\n")
    print("Check: Monte Carlo against the exact mixture over the 2^5 regime paths")
    print(pd.DataFrame(check).to_string(index=False, float_format="{:.3f}".format) + "\n")
    return df, losses, lgd


# ---------------------------------------------------------------- Plots

def plot_results(P_exp, P_rec, P_cm, yearly, losses, lgd):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Yearly default rates, and recession against expansion by rating
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["C3" if r else "C0" for r in yearly["Regime"]]
    axes[0].bar(yearly["Year"], yearly["Spec grade %"], color=colors)
    axes[0].set_xticks(yearly["Year"])
    axes[0].set_xticklabels(yearly["Year"], rotation=45)
    axes[0].set_ylabel("Default rate (%)")
    axes[0].set_title("Speculative-grade default rate (red = recession)")
    r = np.arange(len(LABELS) - 1)
    for j, (name, P, c) in enumerate((("Expansion", P_exp, "C0"), ("Recession", P_rec, "C3"),
                                      ("CreditMetrics 1997", P_cm, "gray"))):
        axes[1].bar(r + (j - 1) * 0.27, 100 * P[:-1, -1], 0.27, color=c, label=name)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-2, 100)
    axes[1].set_xticks(r)
    axes[1].set_xticklabels(LABELS[:-1])
    axes[1].set_ylabel("One-year default probability (%)")
    axes[1].set_title("Default probability by rating")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fitch_default_rates.png"), dpi=150)

    # Loss distributions under each case
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, loss in losses.items():
        k = np.bincount(np.round(loss / lgd).astype(int))
        x = np.arange(len(k)) * lgd
        ax.semilogy(x[k > 0], k[k > 0] / len(loss), "o-", ms=2, lw=1, label=name)
    ax.set_xlabel(f"Loss over {YEARS} years ($M)")
    ax.set_ylabel("Probability")
    ax.set_title("Portfolio loss with Fitch matrices and recession regimes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fitch_regime_losses.png"), dpi=150)
    plt.close("all")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    rng = np.random.default_rng(seed=42)
    P_pool, P_exp, P_rec, P_cm, yearly = part1()
    df, losses, lgd = part2(P_pool, P_exp, P_rec, rng)
    plot_results(P_exp, P_rec, P_cm, yearly, losses, lgd)
    print(f"Plots saved to {PLOT_DIR}")
