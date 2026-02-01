import sys
import subprocess
import os
import argparse

# ---------- DEFENSIVE DEPENDENCY BOOTSTRAP ----------
try:
    import pandas as pd
except ModuleNotFoundError:
    print("[INFO] pandas not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd
# ---------------------------------------------------


UNIVERSES = [
    "tech_core",
    "semis",
    "software",
    "financials",
    "healthcare",
    "discretionary",
]

GRADE_ORDER = {"A": 2, "B+": 1}


def load_stable_files(results_dir):
    frames = []

    for universe in UNIVERSES:
        path = os.path.join(
            results_dir, universe, f"{universe}_stable.csv"
        )

        if not os.path.exists(path):
            print(f"[WARN] Missing: {path}")
            continue

        df = pd.read_csv(path)
        df["universe"] = universe
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_optional(results_dir, filename):
    frames = []

    for universe in UNIVERSES:
        path = os.path.join(results_dir, universe, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["universe"] = universe
            frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--topk", type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    combined = load_stable_files(args.results_dir)

    if combined.empty:
        print("ℹ️  No *_stable.csv found in any universe.")
        print("ℹ️  This is normal during early persistence buildup.")
        print("✅ Aggregation skipped gracefully.")
        return

    combined["grade_rank"] = combined["grade"].map(GRADE_ORDER).fillna(0)

    combined_sorted = combined.sort_values(
        by=["grade_rank", "score_w"],
        ascending=[False, False],
    )

    combined_sorted.drop(columns=["grade_rank"]).to_csv(
        os.path.join(args.out_dir, "combined_pairs_scored.csv"),
        index=False,
    )

    combined_sorted.head(args.topk).drop(
        columns=["grade_rank"]
    ).to_csv(
        os.path.join(args.out_dir, "combined_top150_watchlist.csv"),
        index=False,
    )

    new_global = load_optional(args.results_dir, "new_candidates.csv")
    dropped_global = load_optional(args.results_dir, "dropped_pairs.csv")

    if not new_global.empty:
        new_global.to_csv(
            os.path.join(args.out_dir, "new_candidates_global.csv"),
            index=False,
        )

    if not dropped_global.empty:
        dropped_global.to_csv(
            os.path.join(args.out_dir, "dropped_pairs_global.csv"),
            index=False,
        )

    print("✅ Global aggregation completed successfully")


if __name__ == "__main__":
    main()
