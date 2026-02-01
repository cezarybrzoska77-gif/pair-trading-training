import os
import glob
import argparse
import pandas as pd


UNIVERSES = [
    "tech_core",
    "semis",
    "software",
    "financials",
    "healthcare",
    "discretionary",
]


GRADE_ORDER = {"A": 2, "B+": 1}


def load_stable_files(base_results_dir):
    frames = []

    for universe in UNIVERSES:
        path = os.path.join(
            base_results_dir, universe, f"{universe}_stable.csv"
        )
        if not os.path.exists(path):
            print(f"[WARN] Missing stable file: {path}")
            continue

        df = pd.read_csv(path)
        df["universe"] = universe
        frames.append(df)

    if not frames:
        raise RuntimeError("No *_stable.csv files found in any universe")

    return pd.concat(frames, ignore_index=True)


def aggregate_new_or_dropped(base_results_dir, filename):
    frames = []

    for universe in UNIVERSES:
        path = os.path.join(base_results_dir, universe, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["universe"] = universe
            frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for combined files",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=150,
        help="Top K pairs for watchlist",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ========= LOAD & MERGE STABLE =========
    combined = load_stable_files(args.results_dir)

    # grade ranking
    combined["grade_rank"] = combined["grade"].map(GRADE_ORDER).fillna(0)

    combined_sorted = combined.sort_values(
        by=["grade_rank", "score_w"],
        ascending=[False, False],
    )

    combined_path = os.path.join(
        args.out_dir, "combined_pairs_scored.csv"
    )
    combined_sorted.drop(columns=["grade_rank"]).to_csv(
        combined_path, index=False
    )

    # ========= TOP WATCHLIST =========
    watchlist = combined_sorted.head(args.topk).drop(
        columns=["grade_rank"]
    )

    watchlist_path = os.path.join(
        args.out_dir, "combined_top150_watchlist.csv"
    )
    watchlist.to_csv(watchlist_path, index=False)

    # ========= NEW & DROPPED =========
    new_global = aggregate_new_or_dropped(
        args.results_dir, "new_candidates.csv"
    )
    dropped_global = aggregate_new_or_dropped(
        args.results_dir, "dropped_pairs.csv"
    )

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

    print("✅ Aggregation complete")
    print(f" - {combined_path}")
    print(f" - {watchlist_path}")


if __name__ == "__main__":
    main()
