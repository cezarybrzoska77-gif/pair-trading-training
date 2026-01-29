#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_pairs_retail.py
Retail mode pairs scanner: correlation + cointegration (Engle-Granger).
Sector: Technology (XLK + VGT + FTEC combined).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGER SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tickers_from_file(filepath: str) -> List[str]:
    """Load tickers from text file (one ticker per line)."""
    logger.info(f"Loading tickers from {filepath}")
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Tickers file not found: {filepath}")
        sys.exit(1)
    
    with open(path, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    
    tickers = sorted(list(set(tickers)))  # Remove duplicates
    logger.info(f"Loaded {len(tickers)} unique tickers")
    return tickers


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    auto_adjust: bool
) -> pd.DataFrame:
    """Download price data via yfinance."""
    logger.info(f"Downloading data for {len(tickers)} tickers from {start_date}")
    logger.info(f"Auto-adjust: {auto_adjust}")
    
    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=auto_adjust,
        threads=True
    )
    
    if df.empty:
        logger.error("No data downloaded. Check tickers and date range.")
        sys.exit(1)
    
    # Extract Adj Close or Close
    if "Adj Close" in df.columns.get_level_values(0):
        prices = df["Adj Close"]
    elif "Close" in df.columns.get_level_values(0):
        prices = df["Close"]
    else:
        logger.error("Neither 'Adj Close' nor 'Close' found in downloaded data.")
        sys.exit(1)
    
    # Drop tickers with all NaN
    prices = prices.dropna(axis=1, how="all")
    logger.info(f"Downloaded {len(prices.columns)} tickers with data")
    logger.info(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    return prices


def compute_returns(prices: pd.DataFrame, use_percent: bool) -> pd.DataFrame:
    """Compute log-returns or percent returns."""
    if use_percent:
        logger.info("Computing percent returns")
        returns = prices.pct_change()
    else:
        logger.info("Computing log-returns")
        returns = np.log(prices / prices.shift(1))
    
    return returns


def compute_correlation_windows(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window_60: int = 60,
    window_90: int = 90
) -> Tuple[float, float, int, int]:
    """
    Compute rolling correlation for last 60 and 90 days.
    Returns: (corr_60, corr_90, obs_60, obs_90)
    Requires >=60% of observations in window.
    """
    ret_a = returns[ticker_a]
    ret_b = returns[ticker_b]
    
    # Last 60 days
    data_60 = pd.concat([ret_a.tail(window_60), ret_b.tail(window_60)], axis=1).dropna()
    obs_60 = len(data_60)
    if obs_60 >= int(0.6 * window_60):
        corr_60 = data_60.corr().iloc[0, 1]
    else:
        corr_60 = np.nan
    
    # Last 90 days
    data_90 = pd.concat([ret_a.tail(window_90), ret_b.tail(window_90)], axis=1).dropna()
    obs_90 = len(data_90)
    if obs_90 >= int(0.6 * window_90):
        corr_90 = data_90.corr().iloc[0, 1]
    else:
        corr_90 = np.nan
    
    return corr_60, corr_90, obs_60, obs_90


def compute_cointegration_best(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookbacks: List[int]
) -> Tuple[float, float, int]:
    """
    Run Engle-Granger cointegration test for multiple lookbacks.
    Returns: (best_stat, best_pvalue, best_lookback)
    """
    price_a = prices[ticker_a]
    price_b = prices[ticker_b]
    
    best_pvalue = 1.0
    best_stat = np.nan
    best_lookback = lookbacks[0]
    
    for lb in lookbacks:
        data = pd.concat([price_a.tail(lb), price_b.tail(lb)], axis=1).dropna()
        if len(data) < int(0.6 * lb):
            continue
        
        try:
            stat, pvalue, _ = coint(data.iloc[:, 0], data.iloc[:, 1])
            if pvalue < best_pvalue:
                best_pvalue = pvalue
                best_stat = stat
                best_lookback = lb
        except Exception:
            continue
    
    return best_stat, best_pvalue, best_lookback


def compute_sample_size(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    max_lookback: int
) -> int:
    """Count common observations in the longest lookback window."""
    price_a = prices[ticker_a]
    price_b = prices[ticker_b]
    data = pd.concat([price_a.tail(max_lookback), price_b.tail(max_lookback)], axis=1).dropna()
    return len(data)


def assign_grade(pvalue: float) -> str:
    """Assign grade based on cointegration p-value."""
    if pvalue <= 0.05:
        return "A"
    elif pvalue <= 0.10:
        return "B"
    else:
        return "C"


# ============================================================================
# MAIN SCANNER
# ============================================================================

def scan_pairs(
    tickers: List[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    coint_lookbacks: List[int],
    min_sample: int
) -> pd.DataFrame:
    """
    Scan all pairs and compute correlation + cointegration metrics.
    """
    n = len(tickers)
    logger.info(f"Scanning {n * (n - 1) // 2} pairs")
    
    results = []
    max_lookback = max(coint_lookbacks)
    
    for i in range(n):
        for j in range(i + 1, n):
            a = tickers[i]
            b = tickers[j]
            
            # Correlation windows
            corr_60, corr_90, obs_60, obs_90 = compute_correlation_windows(
                returns, a, b
            )
            
            if np.isnan(corr_60) or np.isnan(corr_90):
                continue
            
            corr_mean = (corr_60 + corr_90) / 2.0
            
            # Cointegration
            coint_stat, coint_pvalue, coint_lb = compute_cointegration_best(
                prices, a, b, coint_lookbacks
            )
            
            # Sample size
            sample = compute_sample_size(prices, a, b, max_lookback)
            
            # Grade
            grade = assign_grade(coint_pvalue)
            
            results.append({
                "a": a,
                "b": b,
                "corr_60": corr_60,
                "corr_90": corr_90,
                "corr_obs_60": obs_60,
                "corr_obs_90": obs_90,
                "corr_mean": corr_mean,
                "coint_stat_best": coint_stat,
                "coint_pvalue_best": coint_pvalue,
                "coint_lookback_best": coint_lb,
                "sample": sample,
                "grade": grade
            })
    
    df = pd.DataFrame(results)
    logger.info(f"Computed metrics for {len(df)} pairs")
    return df


def apply_filters(
    df: pd.DataFrame,
    min_sample: int
) -> pd.DataFrame:
    """
    Apply retail mode filters:
    - Correlation: (mean >= 0.82) OR (corr_60 >= 0.84 AND corr_90 >= 0.80)
    - Cointegration: p-value <= 0.10
    - Sample: >= min_sample
    """
    logger.info("Applying filters...")
    
    corr_pass = (
        (df["corr_mean"] >= 0.82) |
        ((df["corr_60"] >= 0.84) & (df["corr_90"] >= 0.80))
    )
    
    coint_pass = df["coint_pvalue_best"] <= 0.10
    sample_pass = df["sample"] >= min_sample
    
    mask = corr_pass & coint_pass & sample_pass
    filtered = df[mask].copy()
    
    logger.info(f"Pairs after filters: {len(filtered)} / {len(df)}")
    return filtered


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by: grade (A before B), coint_pvalue (asc), corr_90 (desc), corr_60 (desc).
    """
    df = df.sort_values(
        by=["grade", "coint_pvalue_best", "corr_90", "corr_60"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)
    
    return df


# ============================================================================
# CLI & MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retail mode pairs scanner: correlation + cointegration"
    )
    
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of tickers (space-separated)"
    )
    parser.add_argument(
        "--tickers-file",
        default="data/tickers_tech.txt",
        help="Path to file with tickers (default: data/tickers_tech.txt)"
    )
    parser.add_argument(
        "--start-date",
        default="2018-01-01",
        help="Start date (default: 2018-01-01)"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (optional)"
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        default=True,
        help="Use auto-adjust for prices (default: ON)"
    )
    parser.add_argument(
        "--no-auto-adjust",
        dest="auto_adjust",
        action="store_false",
        help="Disable auto-adjust"
    )
    parser.add_argument(
        "--use-percent-returns",
        action="store_true",
        help="Use percent returns instead of log-returns"
    )
    parser.add_argument(
        "--coint-lookbacks",
        default="240,300",
        help="Cointegration lookback windows (comma-separated, default: 240,300)"
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=200,
        help="Minimum sample size (default: 200)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Top K candidates to save (default: 50)"
    )
    parser.add_argument(
        "--out-csv",
        default="results/tech_pairs_candidates.csv",
        help="Output CSV path (default: results/tech_pairs_candidates.csv)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_tickers_from_file(args.tickers_file)
    
    # Parse cointegration lookbacks
    coint_lookbacks = [int(x.strip()) for x in args.coint_lookbacks.split(",")]
    logger.info(f"Cointegration lookbacks: {coint_lookbacks}")
    
    # Download data
    prices = download_data(tickers, args.start_date, args.end_date, args.auto_adjust)
    
    # Compute returns
    returns = compute_returns(prices, args.use_percent_returns)
    
    # Get tickers with valid data
    valid_tickers = sorted(prices.columns.tolist())
    
    # Scan pairs
    all_metrics = scan_pairs(valid_tickers, prices, returns, coint_lookbacks, args.min_sample)
    
    # Save all metrics
    out_all = Path(args.out_csv).parent / "tech_pairs_all_metrics.csv"
    out_all.parent.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(out_all, index=False)
    logger.info(f"Saved all metrics: {out_all}")
    
    # Apply filters
    candidates = apply_filters(all_metrics, args.min_sample)
    
    if len(candidates) == 0:
        logger.warning("No pairs passed filters!")
        # Still create empty CSV
        candidates.to_csv(args.out_csv, index=False)
        logger.info(f"Saved empty candidates CSV: {args.out_csv}")
        return
    
    # Sort candidates
    candidates = sort_candidates(candidates)
    
    # Limit to top K
    if len(candidates) > args.topk:
        candidates = candidates.head(args.topk)
    
    # Save candidates
    candidates.to_csv(args.out_csv, index=False)
    logger.info(f"Saved {len(candidates)} candidates: {args.out_csv}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pairs scanned: {len(all_metrics)}")
    logger.info(f"Pairs passing filters: {len(candidates)}")
    if len(candidates) > 0:
        logger.info(f"Grade A: {len(candidates[candidates['grade'] == 'A'])}")
        logger.info(f"Grade B: {len(candidates[candidates['grade'] == 'B'])}")
        logger.info("\nTop 5 candidates:")
        print(candidates.head(5)[["a", "b", "corr_mean", "coint_pvalue_best", "grade"]])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()