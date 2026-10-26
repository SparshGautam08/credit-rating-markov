# Estimates one-year rating transition counts from Fitch's corporate rating
# history, published under SEC Rule 17g-7(b) and converted to CSV by
# ratingshistory.info. Writes the counts by cohort year to data/, so the main
# script runs without the 29 MB raw file.
#
# Cohort method (the one the rating agencies use in their default studies):
# for each year, take every issuer rated on January 1, look at its rating one
# year later, and count the move. An issuer that defaults at any time during
# the year counts as a move to D, even if it is re-rated later that year (for
# example after a distressed exchange). An issuer whose rating is withdrawn
# without a default leaves the cohort, which is the standard "withdrawal
# adjusted" convention.
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "data", "raw", "fitch_corporate_20261001.csv")
OUT_PATH = os.path.join(HERE, "data", "fitch_transition_counts.csv")
LABELS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
# Ratings outstanding on 2012-06-15 are reported from that date, and the file
# runs to 2025-09-30, so the full one-year cohorts are 2013 to 2024
FIRST_COHORT, LAST_COHORT = 2013, 2024


# Fitch's 22 notches collapsed to the 7 grades plus default. CCC+ to C are
# pooled as CCC, as in the CreditMetrics matrix. RD (restricted default, such
# as a distressed debt exchange) and D are both default. WD and NR mean the
# rating was withdrawn.
def grade(rating):
    if rating in ("RD", "D"):
        return "D"
    if rating in ("WD", "NR"):
        return "WD"
    if rating in ("CC", "C") or rating.startswith("CCC"):
        return "CCC"
    return rating.rstrip("+-")


def load_actions(path):
    df = pd.read_csv(path, dtype=str)
    df = df[df.rating_type == "Long Term Issuer Default Rating"]
    out = pd.DataFrame({
        "issuer": df.obligor_name,
        "date": pd.to_datetime(df.rating_action_date),
        "grade": df.rating.map(grade),
    })
    # Same-day actions keep their order in the file (e.g. RD then CCC on the
    # day a distressed exchange completes)
    return out.sort_values(["issuer", "date"], kind="stable").reset_index(drop=True)


# Counts the cohort transitions for one year. `actions` is grouped by issuer.
def cohort_counts(groups, year):
    start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    counts = np.zeros((len(LABELS) - 1, len(LABELS)), dtype=int)
    withdrawn = np.zeros(len(LABELS) - 1, dtype=int)
    for _, a in groups:
        before = a[a.date <= start]
        if before.empty or before.grade.iloc[-1] in ("D", "WD"):
            continue                                  # not rated, or not in a live rating, on Jan 1
        i = LABELS.index(before.grade.iloc[-1])
        during = a[(a.date > start) & (a.date <= end)]
        if (during.grade == "D").any():
            counts[i, -1] += 1
            continue
        g = a[a.date <= end].grade.iloc[-1]
        if g == "WD":
            withdrawn[i] += 1
        else:
            counts[i, LABELS.index(g)] += 1
    return counts, withdrawn


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RAW_PATH
    actions = load_actions(path)
    print(f"{len(actions):,} issuer default rating actions for {actions.issuer.nunique():,} issuers, "
          f"{actions.date.min():%Y-%m-%d} to {actions.date.max():%Y-%m-%d}")
    groups = list(actions.groupby("issuer"))
    rows = []
    for year in range(FIRST_COHORT, LAST_COHORT + 1):
        counts, withdrawn = cohort_counts(groups, year)
        for i in range(len(LABELS) - 1):
            rows.append({"Year": year, "From": LABELS[i], **dict(zip(LABELS, counts[i])),
                         "Withdrawn": withdrawn[i]})
        print(f"{year}: {counts.sum() + withdrawn.sum():,} issuers on Jan 1, {counts[:, -1].sum()} defaults, "
              f"{withdrawn.sum()} withdrawn")
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")
