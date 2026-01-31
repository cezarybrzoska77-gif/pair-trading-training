#!/usr/bin/env python3
"""
Retail Mode Scanner - Single Basket (Discretionary/XLY)
Calculates correlations, Spearman, residual correlation, cointegration, and scoring.
Outputs two CSV files: all metrics and filtered candidates (A/B+).
"""

import argparse
import os
import sys
from datetime import datetime
from itertools import combinations
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan pairs for single basket with correlations, cointegration, and scoring"
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default="data/tickers_discretionary.txt",
        help="Path to file with ticker symbols (one per line)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Start date for historical data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        default=True,
        help="Use auto-adjusted close prices (default: True)",
    )
    parser.add_argument(
        "--no-auto-adjust",
        action="store_false",
        dest="auto_adjust",
        help="Disable auto-adjustment",
    )
    parser.add_argument(
        "--use-percent-returns",
        action="store_true",
        default=False,
        help="Use percent returns instead of log returns",
    )
    parser.add_argument(
        "--winsorize",
        action="store_true",
        default=True,
        help="Apply winsorization at 1%/99% on returns (default: True)",
    )
    parser.add_argument(
        "--no-winsorize",
        action="store_false",
        dest="winsorize",
        help="Disable winsorization",
    )
    parser.add_argument(
        "--coint-lookbacks",
        type=str,
        default="240,300",
        help="Comma-separated cointegration lookback periods (default: 240,300)",
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=200,
        help="Minimum number of common observations required (default: 200)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Maximum number of top pairs to process (default: 100)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/discretionary",
        help="Output directory for results (default: results/discretionary)",
    )
    return parser.parse_args()


def load_tickers(filepath: str) -> List[str]:
    """Load ticker symbols from file (one per line)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tickers file not found: {filepath}")
    
    with open(filepath, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)
    
    print(f"Loaded {len(unique_tickers)} unique tickers from {filepath}")
    return unique_tickers


def download_prices(tickers: List[str], start_date: str, auto_adjust: bool) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    print(f"Downloading price data from {start_date}...")
    
    data = yf.download(
        tickers,
        start=start_date,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True
    )
    
    if auto_adjust:
        prices = data["Close"] if len(tickers) > 1 else data[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        prices = data["Adj Close"] if len(tickers) > 1 else data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    
    # Ensure we have a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    # Drop columns with all NaN
    prices = prices.dropna(axis=1, how="all")
    
    print(f"Downloaded data for {len(prices.columns)} tickers")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    return prices


def calculate_returns(prices: pd.DataFrame, use_percent: bool, winsorize: bool) -> pd.DataFrame:
    """Calculate returns (log or percent) with optional winsorization."""
    if use_percent:
        returns = prices.pct_change()
    else:
        returns = np.log(prices / prices.shift(1))
    
    returns = returns.dropna(how="all")
    
    if winsorize:
        # Winsorize at 1% and 99% percentiles
        for col in returns.columns:
            lower = returns[col].quantile(0.01)
            upper = returns[col].quantile(0.99)
            returns[col] = returns[col].clip(lower=lower, upper=upper)
    
    return returns


def calculate_correlation_metrics(
    ret1: pd.Series,
    ret2: pd.Series,
    prices1: pd.Series,
    prices2: pd.Series,
    index_prices: pd.Series
) -> Dict[str, Any]:
    """Calculate all correlation-based metrics for a pair."""
    metrics = {}
    
    # Align data
    common_idx = ret1.dropna().index.intersection(ret2.dropna().index)
    r1 = ret1.loc[common_idx]
    r2 = ret2.loc[common_idx]
    p1 = prices1.loc[common_idx]
    p2 = prices2.loc[common_idx]
    
    if len(r1) < 60:
        return None
    
    metrics["sample"] = len(r1)
    
    # Pearson correlations
    if len(r1) >= 60:
        metrics["corr_60"] = r1.tail(60).corr(r2.tail(60))
        metrics["corr_obs_60"] = 60
    else:
        metrics["corr_60"] = np.nan
        metrics["corr_obs_60"] = 0
    
    if len(r1) >= 90:
        metrics["corr_90"] = r1.tail(90).corr(r2.tail(90))
        metrics["corr_obs_90"] = 90
    else:
        metrics["corr_90"] = np.nan
        metrics["corr_obs_90"] = 0
    
    metrics["corr_mean"] = (metrics["corr_60"] + metrics["corr_90"]) / 2 if not np.isnan(metrics["corr_60"]) and not np.isnan(metrics["corr_90"]) else np.nan
    
    # Spearman correlations
    if len(r1) >= 60:
        metrics["spearman_60"], _ = stats.spearmanr(r1.tail(60), r2.tail(60))
    else:
        metrics["spearman_60"] = np.nan
    
    if len(r1) >= 90:
        metrics["spearman_90"], _ = stats.spearmanr(r1.tail(90), r2.tail(90))
    else:
        metrics["spearman_90"] = np.nan
    
    # Residual correlation after index regression
    idx_common = common_idx.intersection(index_prices.dropna().index)
    if len(idx_common) >= 60:
        p1_idx = p1.loc[idx_common]
        p2_idx = p2.loc[idx_common]
        idx_p = index_prices.loc[idx_common]
        
        # Regression: P1 ~ alpha + beta * Index
        X1 = np.column_stack([np.ones(len(idx_p)), idx_p.values])
        beta1 = np.linalg.lstsq(X1, p1_idx.values, rcond=None)[0]
        resid1 = p1_idx.values - X1 @ beta1
        
        # Regression: P2 ~ alpha + beta * Index
        beta2 = np.linalg.lstsq(X1, p2_idx.values, rcond=None)[0]
        resid2 = p2_idx.values - X1 @ beta2
        
        # Correlation of residuals
        if len(resid1) >= 60:
            metrics["resid_corr_60"] = np.corrcoef(resid1[-60:], resid2[-60:])[0, 1]
        else:
            metrics["resid_corr_60"] = np.nan
        
        if len(resid1) >= 90:
            metrics["resid_corr_90"] = np.corrcoef(resid1[-90:], resid2[-90:])[0, 1]
        else:
            metrics["resid_corr_90"] = np.nan
    else:
        metrics["resid_corr_60"] = np.nan
        metrics["resid_corr_90"] = np.nan
    
    # Correlation hitrate (30-day windows over 6 months ≈ 126 days)
    if len(r1) >= 126:
        recent_r1 = r1.tail(126)
        recent_r2 = r2.tail(126)
        
        window_size = 30
        hitrate_count = 0
        total_windows = 0
        
        for i in range(len(recent_r1) - window_size + 1):
            window_r1 = recent_r1.iloc[i:i+window_size]
            window_r2 = recent_r2.iloc[i:i+window_size]
            
            if len(window_r1) == window_size and len(window_r2) == window_size:
                corr_window = window_r1.corr(window_r2)
                if corr_window >= 0.80:
                    hitrate_count += 1
                total_windows += 1
        
        metrics["corr_hitrate_30d_6m"] = hitrate_count / total_windows if total_windows > 0 else 0.0
    else:
        metrics["corr_hitrate_30d_6m"] = np.nan
    
    return metrics


def calculate_cointegration(
    prices1: pd.Series,
    prices2: pd.Series,
    lookbacks: List[int]
) -> Tuple[float, float, int]:
    """Calculate Engle-Granger cointegration for multiple lookbacks and return best."""
    common_idx = prices1.dropna().index.intersection(prices2.dropna().index)
    p1 = prices1.loc[common_idx]
    p2 = prices2.loc[common_idx]
    
    best_pvalue = 1.0
    best_stat = 0.0
    best_lookback = 0
    
    for lookback in lookbacks:
        if len(p1) >= lookback:
            p1_window = p1.tail(lookback)
            p2_window = p2.tail(lookback)
            
            try:
                stat, pvalue, _ = coint(p1_window, p2_window)
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_stat = stat
                    best_lookback = lookback
            except Exception:
                continue
    
    return best_pvalue, best_stat, best_lookback


def apply_filters(row: pd.Series) -> str:
    """Apply A/B+ filters to determine grade."""
    # Grade A criteria
    if (
        row["corr_mean"] >= 0.82
        and row["coint_pvalue_best"] <= 0.05
        and row["corr_hitrate_30d_6m"] >= 0.70
        and max(row["resid_corr_60"], row["resid_corr_90"]) >= 0.60
        and row["sample"] >= 200
    ):
        return "A"
    
    # Grade B+ criteria
    resid_max = max(row["resid_corr_60"], row["resid_corr_90"])
    corr_condition = row["corr_mean"] >= 0.78 or (row["corr_60"] >= 0.80 and row["corr_90"] >= 0.76)
    
    if (
        corr_condition
        and row["coint_pvalue_best"] <= 0.12
        and row["spearman_60"] >= 0.75
        and row["spearman_90"] >= 0.75
        and resid_max >= 0.60
        and row["corr_hitrate_30d_6m"] >= 0.70
        and row["sample"] >= 200
    ):
        return "B+"
    
    return "None"


def calculate_score(row: pd.Series) -> float:
    """Calculate weighted score (0-1) with grade bonuses."""
    # Component scores (normalized to 0-1)
    corr_mean_score = max(0, min(1, row["corr_mean"])) if not np.isnan(row["corr_mean"]) else 0
    
    spearman_mean = (row["spearman_60"] + row["spearman_90"]) / 2
    spearman_score = max(0, min(1, spearman_mean)) if not np.isnan(spearman_mean) else 0
    
    resid_corr_max = max(row["resid_corr_60"], row["resid_corr_90"])
    resid_score = max(0, min(1, resid_corr_max)) if not np.isnan(resid_corr_max) else 0
    
    # Cointegration: lower p-value is better, normalize inversely
    coint_score = max(0, 1 - row["coint_pvalue_best"]) if not np.isnan(row["coint_pvalue_best"]) else 0
    
    hitrate_score = max(0, min(1, row["corr_hitrate_30d_6m"])) if not np.isnan(row["corr_hitrate_30d_6m"]) else 0
    
    # Weighted score
    score = (
        0.30 * corr_mean_score +
        0.15 * spearman_score +
        0.20 * resid_score +
        0.20 * coint_score +
        0.10 * hitrate_score
    )
    
    # Grade bonuses
    if row["grade"] == "A":
        score += 0.03
    elif row["grade"] == "B+":
        score += 0.01
    
    return min(1.0, score)


def scan_pairs(
    tickers: List[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    index_ticker: str,
    coint_lookbacks: List[int],
    min_sample: int,
    topk: int
) -> pd.DataFrame:
    """Scan all pairs and calculate metrics."""
    # Download index data
    print(f"Downloading index data ({index_ticker})...")
    index_data = yf.download(index_ticker, start=prices.index[0], auto_adjust=True, progress=False)
    index_prices = index_data["Close"] if "Close" in index_data.columns else index_data
    
    results = []
    total_pairs = len(list(combinations(tickers, 2)))
    
    print(f"Processing {total_pairs} pairs...")
    
    for i, (tick1, tick2) in enumerate(combinations(tickers, 2)):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_pairs} pairs...")
        
        if tick1 not in prices.columns or tick2 not in prices.columns:
            continue
        
        if tick1 not in returns.columns or tick2 not in returns.columns:
            continue
        
        # Calculate correlation metrics
        corr_metrics = calculate_correlation_metrics(
            returns[tick1],
            returns[tick2],
            prices[tick1],
            prices[tick2],
            index_prices
        )
        
        if corr_metrics is None:
            continue
        
        if corr_metrics["sample"] < min_sample:
            continue
        
        # Calculate cointegration
        coint_pvalue, coint_stat, coint_lookback = calculate_cointegration(
            prices[tick1],
            prices[tick2],
            coint_lookbacks
        )
        
        # Combine all metrics
        pair_result = {
            "ticker1": tick1,
            "ticker2": tick2,
            **corr_metrics,
            "coint_pvalue_best": coint_pvalue,
            "coint_stat_best": coint_stat,
            "coint_lookback_best": coint_lookback,
        }
        
        results.append(pair_result)
    
    print(f"Completed processing {len(results)} valid pairs")
    
    if not results:
        print("WARNING: No valid pairs found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Apply filters
    df["grade"] = df.apply(apply_filters, axis=1)
    
    # Calculate scores
    df["score_w"] = df.apply(calculate_score, axis=1)
    
    return df


def main():
    """Main execution function."""
    args = parse_args()
    
    # Parse cointegration lookbacks
    coint_lookbacks = [int(x.strip()) for x in args.coint_lookbacks.split(",")]
    
    # Load tickers
    tickers = load_tickers(args.tickers_file)
    
    # Download prices
    prices = download_prices(tickers, args.start_date, args.auto_adjust)
    
    # Calculate returns
    returns = calculate_returns(prices, args.use_percent_returns, args.winsorize)
    
    # Determine index ticker (hardcoded for discretionary basket)
    index_ticker = "XLY"
    
    # Scan pairs
    results_df = scan_pairs(
        list(prices.columns),
        prices,
        returns,
        index_ticker,
        coint_lookbacks,
        args.min_sample,
        args.topk
    )
    
    if results_df.empty:
        print("No results to save. Exiting.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save all metrics
    all_metrics_path = os.path.join(args.out_dir, "discretionary_all_metrics.csv")
    results_df.to_csv(all_metrics_path, index=False)
    print(f"Saved all metrics to {all_metrics_path}")
    
    # Filter candidates (A and B+ only)
    candidates_df = results_df[results_df["grade"].isin(["A", "B+"])].copy()
    
    if not candidates_df.empty:
        # Sort: grade (A before B+), pvalue asc, corr_90 desc, corr_60 desc
        grade_order = {"A": 0, "B+": 1}
        candidates_df["grade_order"] = candidates_df["grade"].map(grade_order)
        candidates_df = candidates_df.sort_values(
            by=["grade_order", "coint_pvalue_best", "corr_90", "corr_60"],
            ascending=[True, True, False, False]
        )
        candidates_df = candidates_df.drop(columns=["grade_order"])
        
        candidates_path = os.path.join(args.out_dir, "discretionary_candidates.csv")
        candidates_df.to_csv(candidates_path, index=False)
        print(f"Saved {len(candidates_df)} candidates to {candidates_path}")
        print(f"  Grade A: {len(candidates_df[candidates_df['grade'] == 'A'])}")
        print(f"  Grade B+: {len(candidates_df[candidates_df['grade'] == 'B+'])}")
    else:
        print("No A or B+ candidates found")
    
    print("Scan complete!")


if __name__ == "__main__":
    main()
```

## 2. `requirements.txt`
```
numpy>=1.24.0
pandas>=2.0.0
statsmodels>=0.14.0
yfinance>=0.2.28
scipy>=1.10.0