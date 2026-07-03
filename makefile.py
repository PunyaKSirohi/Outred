#!/usr/bin/env python3
"""
generate_test_csv.py

Generates a clean, well-formed CSV (habitation-style schema) with a
controlled contamination rate of statistical outliers, for testing
Outred's detection engines (IForest, HBOS, LOF, CBLOF, OCSVM, ensemble,
incremental SGDOneClassSVM) against a known ground truth.

This has nothing to do with CSV structural repair -- every row here is
valid RFC4180. The "contamination" is purely statistical:

  - Numerical outliers: values far outside the normal distribution for
    their column (extreme magnitude, negative-where-only-positive-makes-
    sense, impossible ratios, zero-inflation, decimal/unit-entry typos).
  - Categorical outliers: rare/invalid category values that don't belong
    to the normal category set for that column (typos, wrong-case
    variants, values from a different domain entirely, singleton
    categories that appear ~once in the whole file).

Each row is tagged with a ground-truth IS_OUTLIER column (1/0) and an
OUTLIER_TYPE column so you can score precision/recall against Outred's
predictions. Use --no-ground-truth for a blind test file.

Usage:
    python3 generate_test_csv.py --rows 2000000 --contamination 0.02 --out test.csv
"""

import argparse
import random
import sys

FIELDS = [
    "STATE", "DISTRICT", "BLOCK", "GP", "VILLAGE", "HABITATION",
    "POP_SC", "POP_ST", "POP_GEN", "HH_SC", "HH_ST", "HH_GEN",
    "STATUS", "FLAG1", "FLAG2", "DATE",
]

GROUND_TRUTH_FIELDS = ["IS_OUTLIER", "OUTLIER_TYPE"]

STATES = [
    "ASSAM", "BIHAR", "GUJARAT", "HARYANA", "KARNATAKA", "MADHYA PRADESH",
    "MEGHALAYA", "MIZORAM", "NAGALAND", "ORISSA", "RAJASTHAN", "TAMIL NADU",
    "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND", "WEST BENGAL", "CHATTISGARH",
    "JHARKHAND", "SIKKIM", "JAMMU AND KASHMIR",
]

STATUSES = ["Fully Covered", "Partially Covered", "Not Covered"]
YES_NO = ["YES", "NO"]
DATES = ["01_04_2009", "01_04_2010", "01_10_2009"]

_SYLLABLES = [
    "MA", "RA", "PU", "LI", "KO", "TA", "GA", "NI", "DA", "SU", "BHA",
    "CHA", "WA", "HA", "JA", "SA", "VI", "THU", "RI", "PA", "KA", "MU",
]


def _random_name(min_syl=2, max_syl=4):
    n = random.randint(min_syl, max_syl)
    return "".join(random.choice(_SYLLABLES) for _ in range(n))


def _random_pool(n, min_syl=2, max_syl=4):
    return [_random_name(min_syl, max_syl) for _ in range(n)]


def build_pools():
    return {
        "districts": _random_pool(150),
        "blocks": _random_pool(600),
        "gps": _random_pool(3000),
        "villages": _random_pool(8000),
        "habitations": _random_pool(12000, 2, 5),
    }


def quote(field) -> str:
    return '"' + str(field).replace('"', '""') + '"'


def random_clean_row(pools):
    """A statistically normal row. Numeric fields drawn from tight, realistic ranges."""
    return {
        "STATE": random.choice(STATES),
        "DISTRICT": random.choice(pools["districts"]),
        "BLOCK": random.choice(pools["blocks"]),
        "GP": random.choice(pools["gps"]),
        "VILLAGE": random.choice(pools["villages"]),
        "HABITATION": random.choice(pools["habitations"]),
        "POP_SC": random.randint(0, 800),
        "POP_ST": random.randint(0, 800),
        "POP_GEN": random.randint(0, 2000),
        "HH_SC": random.randint(0, 300),
        "HH_ST": random.randint(0, 300),
        "HH_GEN": random.randint(0, 900),
        "STATUS": random.choice(STATUSES),
        "FLAG1": random.choice(YES_NO),
        "FLAG2": random.choice(YES_NO),
        "DATE": random.choice(DATES),
    }


# ---------------------------------------------------------------------------
# Outlier injectors. Each mutates a normal row dict in place to become a
# genuine statistical outlier and returns a short ground-truth label.
# All output stays valid CSV -- only the VALUES are anomalous, not the syntax.
# ---------------------------------------------------------------------------

def outlier_numeric_extreme_high(row):
    """Magnitude outlier: one numeric column blown up 50-500x normal range."""
    col = random.choice(["POP_SC", "POP_ST", "POP_GEN", "HH_SC", "HH_ST", "HH_GEN"])
    row[col] = random.randint(50_000, 500_000)
    return f"numeric_extreme_high:{col}"


def outlier_numeric_negative(row):
    """Sign outlier: negative value where only non-negative counts make sense."""
    col = random.choice(["POP_SC", "POP_ST", "POP_GEN", "HH_SC", "HH_ST", "HH_GEN"])
    row[col] = -random.randint(1, 500)
    return f"numeric_negative:{col}"


def outlier_numeric_impossible_ratio(row):
    """Consistency outlier: households exceed population for the same group."""
    group = random.choice(["SC", "ST", "GEN"])
    pop_col, hh_col = f"POP_{group}", f"HH_{group}"
    row[pop_col] = random.randint(1, 20)
    row[hh_col] = row[pop_col] + random.randint(200, 2000)  # more households than people
    return f"numeric_impossible_ratio:{pop_col}/{hh_col}"


def outlier_numeric_zero_cluster(row):
    """All six numeric fields simultaneously zero -- implausible for a real habitation."""
    for col in ["POP_SC", "POP_ST", "POP_GEN", "HH_SC", "HH_ST", "HH_GEN"]:
        row[col] = 0
    return "numeric_zero_cluster"


def outlier_numeric_micro_decimal_typo(row):
    """Opposite-direction magnitude outlier: a plausible value shrunk by ~3 orders
    of magnitude, simulating a decimal/unit entry error."""
    col = random.choice(["POP_GEN", "HH_GEN"])
    row[col] = round(random.uniform(0.001, 0.09), 4)
    return f"numeric_micro_decimal_typo:{col}"


def outlier_categorical_invalid_status(row):
    """Category outlier: STATUS value outside the known {Fully/Partially/Not Covered} set."""
    bogus = random.choice([
        "COVERD", "fully covered", "PENDING", "N/A", "Covered-Partial", "UNKNOWN", "###",
    ])
    row["STATUS"] = bogus
    return "categorical_invalid_status"


def outlier_categorical_invalid_flag(row):
    """Category outlier: FLAG1/FLAG2 value outside the known {YES, NO} set."""
    col = random.choice(["FLAG1", "FLAG2"])
    bogus = random.choice(["Y", "N", "TRUE", "FALSE", "yes", "1", "0", "MAYBE"])
    row[col] = bogus
    return f"categorical_invalid_flag:{col}"


def outlier_categorical_unknown_state(row):
    """Category outlier: STATE value from outside the known 20-state set entirely."""
    bogus = random.choice([
        "ATLANTIS", "TESTSTATE", "UNKNOWN", "XXXXXXX", "PUNJAB_TYPO", "N/A",
    ])
    row["STATE"] = bogus
    return "categorical_unknown_state"


def outlier_categorical_rare_singleton(row, salt):
    """Category outlier: a category value that appears essentially once in the whole
    file -- a singleton rare level Outred's categorical engine should flag."""
    row["DISTRICT"] = f"RAREDIST_{salt}"
    return "categorical_rare_singleton:DISTRICT"


NUMERIC_INJECTORS = [
    outlier_numeric_extreme_high,
    outlier_numeric_negative,
    outlier_numeric_impossible_ratio,
    outlier_numeric_zero_cluster,
    outlier_numeric_micro_decimal_typo,
]

CATEGORICAL_INJECTORS = [
    outlier_categorical_invalid_status,
    outlier_categorical_invalid_flag,
    outlier_categorical_unknown_state,
]


def render_line(row, include_ground_truth, is_outlier, outlier_type):
    values = [row[f] for f in FIELDS]
    parts = [quote(v) for v in values]
    if include_ground_truth:
        parts.append(quote(1 if is_outlier else 0))
        parts.append(quote(outlier_type or ""))
    return ",".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--contamination", type=float, default=0.02,
                     help="Fraction of rows that are outliers (default 0.02 = 2%%).")
    ap.add_argument("--numeric-share", type=float, default=0.5,
                     help="Of the contaminated rows, fraction that are numeric-type "
                          "outliers vs categorical-type (default 0.5 = even split).")
    ap.add_argument("--singleton-share", type=float, default=0.1,
                     help="Of the categorical outliers, fraction that are rare "
                          "singleton categories rather than invalid values.")
    ap.add_argument("--out", type=str, default="test_habitation.csv")
    ap.add_argument("--include-ground-truth", action="store_true", default=True)
    ap.add_argument("--no-ground-truth", dest="include_ground_truth", action="store_false",
                     help="Omit IS_OUTLIER/OUTLIER_TYPE columns for a blind test file.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    pools = build_pools()

    n_contam = int(args.rows * args.contamination)
    n_numeric = int(n_contam * args.numeric_share)
    n_categorical = n_contam - n_numeric
    n_singleton = int(n_categorical * args.singleton_share)
    n_categorical_invalid = n_categorical - n_singleton

    all_idx = list(range(args.rows))
    random.shuffle(all_idx)
    numeric_idx = set(all_idx[:n_numeric])
    categorical_invalid_idx = set(all_idx[n_numeric:n_numeric + n_categorical_invalid])
    singleton_idx = set(all_idx[n_numeric + n_categorical_invalid:n_contam])

    stats = {"clean": 0, "numeric_outlier": 0, "categorical_outlier": 0}
    type_counts = {}

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        header = list(FIELDS)
        if args.include_ground_truth:
            header += GROUND_TRUTH_FIELDS
        f.write(",".join(quote(h) for h in header) + "\n")

        for i in range(args.rows):
            row = random_clean_row(pools)
            is_outlier = False
            outlier_type = None

            if i in numeric_idx:
                injector = random.choice(NUMERIC_INJECTORS)
                outlier_type = injector(row)
                is_outlier = True
                stats["numeric_outlier"] += 1
            elif i in categorical_invalid_idx:
                injector = random.choice(CATEGORICAL_INJECTORS)
                outlier_type = injector(row)
                is_outlier = True
                stats["categorical_outlier"] += 1
            elif i in singleton_idx:
                outlier_type = outlier_categorical_rare_singleton(row, salt=i)
                is_outlier = True
                stats["categorical_outlier"] += 1
            else:
                stats["clean"] += 1

            if outlier_type:
                key = outlier_type.split(":")[0]
                type_counts[key] = type_counts.get(key, 0) + 1

            f.write(render_line(row, args.include_ground_truth, is_outlier, outlier_type) + "\n")

            if (i + 1) % 200_000 == 0:
                print(f"  ... {i + 1:,} / {args.rows:,} rows written", file=sys.stderr)

    total_outliers = stats["numeric_outlier"] + stats["categorical_outlier"]
    print(f"Done -> {args.out}")
    print(f"  clean rows:             {stats['clean']:,}")
    print(f"  numeric outliers:       {stats['numeric_outlier']:,}")
    print(f"  categorical outliers:   {stats['categorical_outlier']:,}")
    print(f"  total contamination:    {total_outliers:,} ({total_outliers / args.rows * 100:.2f}%)")
    print("  breakdown by type:")
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {t:35s} {c:,}")


if __name__ == "__main__":
    main()