#!/usr/bin/env python3
"""
scan_pairs_retail.py
Retail-mode pairs scanner (Stage 1: Correlation + Cointegration + Quality Gates)
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


def log_info(msg: str):
    """Print info log with prefix."""
    print(f"[INFO] {msg}")


def load_tickers_from_file(path: str) -> List[str]:
    """Load tickers from text file, one per line, skip comments and duplicates."""
    log_info(f"Loading tickers from {path}")
    tickers = []
    with open(path, "r") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line and line not in tickers:
                tickers.append(line)
    log_info(f"Loaded {len(tickers)} unique tickers")
    return tickers


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    auto_adjust: bool
) -> pd.DataFrame:
    """Download Adj Close data via yfinance."""
    log_info(f"Downloading data from {start_date} to {end_date or 'now'}, auto_adjust={auto_adjust}")
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        show_errors=False
    )
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        prices = data
    
    prices = prices.dropna(how="all", axis=0).dropna(how="all", axis=1)
    log_info(f"Downloaded data shape: {prices.shape}")
    return prices


def compute_returns(prices: pd.DataFrame, use_percent: bool) -> pd.DataFrame:
    """Compute returns: log or percent."""
    if use_percent:
        log_info("Computing percent returns (pct_change)")
        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        log_info("Computing log returns")
        returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    return returns


def rolling_correlation(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int,
    min_obs_frac: float = 0.6
) -> Tuple[float, int]:
    """
    Compute rolling Pearson correlation on last 'window' observations.
    Returns (corr_value, valid_obs_count).
    If valid_obs < min_obs_frac * window, return (NaN, 0).
    """
    ret_a = returns[ticker_a].iloc[-window:]
    ret_b = returns[ticker_b].iloc[-window:]
    combined = pd.concat([ret_a, ret_b], axis=1).dropna()
    valid_obs = len(combined)
    if valid_obs < min_obs_frac * window:
        return (np.nan, 0)
    if valid_obs < 2:
        return (np.nan, 0)
    corr_val = combined.iloc[:, 0].corr(combined.iloc[:, 1])
    return (corr_val, valid_obs)


def rolling_spearman(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int,
    min_obs_frac: float = 0.6
) -> float:
    """Compute rolling Spearman correlation on last 'window' observations."""
    ret_a = returns[ticker_a].iloc[-window:]
    ret_b = returns[ticker_b].iloc[-window:]
    combined = pd.concat([ret_a, ret_b], axis=1).dropna()
    valid_obs = len(combined)
    if valid_obs < min_obs_frac * window:
        return np.nan
    if valid_obs < 3:
        return np.nan
    rho, _ = spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    return rho


def residual_correlation(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    index_ticker: str,
    window: int
) -> float:
    """
    Regress each ticker on index (OLS), compute correlation of residuals.
    Returns residual_corr for last 'window' observations.
    """
    if index_ticker not in prices.columns:
        return np.nan
    sub = prices[[ticker_a, ticker_b, index_ticker]].iloc[-window:].dropna()
    if len(sub) < 30:
        return np.nan
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    X = add_constant(sub[index_ticker])
    
    try:
        model_a = OLS(sub[ticker_a], X).fit()
        resid_a = model_a.resid
        model_b = OLS(sub[ticker_b], X).fit()
        resid_b = model_b.resid
        res_corr = resid_a.corr(resid_b)
        return res_corr
    except Exception:
        return np.nan


def engle_granger_test(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback: int
) -> Tuple[float, float]:
    """
    Engle-Granger cointegration test on last 'lookback' observations.
    Returns (coint_stat, pvalue).
    """
    sub = prices[[ticker_a, ticker_b]].iloc[-lookback:].dropna()
    if len(sub) < 50:
        return (np.nan, np.nan)
    try:
        stat, pval, _ = coint(sub[ticker_a], sub[ticker_b])
        return (stat, pval)
    except Exception:
        return (np.nan, np.nan)


def best_cointegration(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookbacks: List[int]
) -> Tuple[float, float, int]:
    """
    Run EG test for multiple lookbacks, return best (lowest p-value).
    Returns (coint_stat_best, coint_pvalue_best, coint_lookback_best).
    """
    results = []
    for lb in lookbacks:
        stat, pval = engle_granger_test(prices, ticker_a, ticker_b, lb)
        if not np.isnan(pval):
            results.append((stat, pval, lb))
    if not results:
        return (np.nan, np.nan, np.nan)
    results.sort(key=lambda x: x[1])
    return results[0]


def compute_sample_size(prices: pd.DataFrame, ticker_a: str, ticker_b: str) -> int:
    """Count common observations for pair over entire period."""
    sub = prices[[ticker_a, ticker_b]].dropna()
    return len(sub)


def scan_all_pairs(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: List[str],
    coint_lookbacks: List[int],
    residual_index: str,
    min_sample: int,
    strict_mode: bool
) -> pd.DataFrame:
    """
    Scan all pairs and compute metrics:
    - corr_60, corr_90, spearman_60, spearman_90
    - resid_corr_60, resid_corr_90
    - coint_stat_best, coint_pvalue_best, coint_lookback_best
    - sample
    - grade (A/B+/B or None)
    Returns DataFrame with all pairs.
    """
    log_info("Scanning all pairs and computing metrics...")
    rows = []
    n = len(tickers)
    total = n * (n - 1) // 2
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            count += 1
            if count % 100 == 0:
                log_info(f"Processed {count}/{total} pairs")
            
            ticker_a = tickers[i]
            ticker_b = tickers[j]
            
            # Correlation
            corr_60, obs_60 = rolling_correlation(returns, ticker_a, ticker_b, 60)
            corr_90, obs_90 = rolling_correlation(returns, ticker_a, ticker_b, 90)
            corr_mean = np.nanmean([corr_60, corr_90])
            
            # Spearman
            spear_60 = rolling_spearman(returns, ticker_a, ticker_b, 60)
            spear_90 = rolling_spearman(returns, ticker_a, ticker_b, 90)
            
            # Residual correlation
            resid_60 = residual_correlation(prices, ticker_a, ticker_b, residual_index, 60)
            resid_90 = residual_correlation(prices, ticker_a, ticker_b, residual_index, 90)
            
            # Cointegration
            coint_stat, coint_pval, coint_lb = best_cointegration(
                prices, ticker_a, ticker_b, coint_lookbacks
            )
            
            # Sample size
            sample = compute_sample_size(prices, ticker_a, ticker_b)
            
            # Grade logic
            grade = None
            
            # Main filters (level 1)
            corr_pass_main = (corr_mean >= 0.82) or (corr_60 >= 0.84 and corr_90 >= 0.80)
            coint_pass = coint_pval <= 0.10 if not np.isnan(coint_pval) else False
            sample_pass = sample >= min_sample
            
            if corr_pass_main and coint_pass and sample_pass:
                if coint_pval <= 0.05:
                    grade = "A"
                else:
                    grade = "B"
            
            # Relaxed mode (B+ with quality gates)
            if not strict_mode and grade is None:
                corr_pass_relax = (corr_mean >= 0.80) or (corr_60 >= 0.82 and corr_90 >= 0.78)
                spear_gate = (spear_60 >= 0.75 and spear_90 >= 0.75) if not (np.isnan(spear_60) or np.isnan(spear_90)) else False
                resid_gate = max(resid_60, resid_90) >= 0.60 if not (np.isnan(resid_60) or np.isnan(resid_90)) else False
                
                if corr_pass_relax and coint_pass and sample_pass and spear_gate and resid_gate:
                    grade = "B+"
            
            rows.append({
                "a": ticker_a,
                "b": ticker_b,
                "corr_60": corr_60,
                "corr_90": corr_90,
                "corr_obs_60": obs_60,
                "corr_obs_90": obs_90,
                "corr_mean": corr_mean,
                "spearman_60": spear_60,
                "spearman_90": spear_90,
                "resid_corr_60": resid_60,
                "resid_corr_90": resid_90,
                "coint_stat_best": coint_stat,
                "coint_pvalue_best": coint_pval,
                "coint_lookback_best": coint_lb,
                "sample": sample,
                "grade": grade
            })
    
    log_info(f"Total pairs scanned: {len(rows)}")
    df = pd.DataFrame(rows)
    return df


def filter_and_sort_candidates(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    """Filter candidates (grade not None) and sort by grade, coint_pvalue, corr."""
    candidates = df[df["grade"].notna()].copy()
    log_info(f"Candidates after filters: {len(candidates)}")
    
    if candidates.empty:
        return candidates
    
    # Sort
    grade_order = {"A": 0, "B+": 1, "B": 2}
    candidates["grade_order"] = candidates["grade"].map(grade_order)
    candidates = candidates.sort_values(
        by=["grade_order", "coint_pvalue_best", "corr_90", "corr_60"],
        ascending=[True, True, False, False]
    ).drop(columns=["grade_order"])
    
    if topk > 0:
        candidates = candidates.head(topk)
    
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Retail-mode pairs scanner (Stage 1: Correlation + Cointegration + Quality Gates)"
    )
    parser.add_argument("--tickers", nargs="+", help="List of tickers")
    parser.add_argument("--tickers-file", default="data/tickers_tech.txt", help="Path to tickers file")
    parser.add_argument("--start-date", default="2018-01-01", help="Start date")
    parser.add_argument("--end-date", help="End date (optional)")
    parser.add_argument("--auto-adjust", dest="auto_adjust", action="store_true", default=True, help="Enable auto-adjust (default)")
    parser.add_argument("--no-auto-adjust", dest="auto_adjust", action="store_false", help="Disable auto-adjust")
    parser.add_argument("--use-percent-returns", action="store_true", help="Use percent returns instead of log returns")
    parser.add_argument("--residual-index", default="QQQ", help="Index ticker for residual correlation")
    parser.add_argument("--coint-lookbacks", default="240,300", help="Cointegration lookback periods (comma-separated)")
    parser.add_argument("--min-sample", type=int, default=200, help="Minimum sample size")
    parser.add_argument("--topk", type=int, default=50, help="Top K candidates to output")
    parser.add_argument("--strict", action="store_true", help="Strict mode (disable B+ relaxation)")
    parser.add_argument("--out-csv", default="results/tech_pairs_candidates.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    # Load tickers
    if args.tickers:
        tickers = args.tickers
    else:
        try:
            tickers = load_tickers_from_file(args.tickers_file)
        except FileNotFoundError:
            log_info(f"ERROR: Tickers file not found: {args.tickers_file}")
            sys.exit(1)
    
    if len(tickers) < 2:
        log_info("ERROR: Need at least 2 tickers")
        sys.exit(1)
    
    # Parse lookbacks
    try:
        coint_lookbacks = [int(x.strip()) for x in args.coint_lookbacks.split(",")]
    except ValueError:
        log_info("ERROR: Invalid --coint-lookbacks format")
        sys.exit(1)
    
    # Add residual index to tickers if not present
    all_tickers = list(set(tickers + [args.residual_index]))
    
    # Download data
    try:
        prices = download_data(all_tickers, args.start_date, args.end_date, args.auto_adjust)
    except Exception as e:
        log_info(f"ERROR: Failed to download data: {e}")
        sys.exit(1)
    
    if prices.empty:
        log_info("ERROR: No data downloaded")
        sys.exit(1)
    
    # Compute returns
    returns = compute_returns(prices, args.use_percent_returns)
    
    # Scan all pairs
    all_metrics = scan_all_pairs(
        prices, returns, tickers, coint_lookbacks, args.residual_index, args.min_sample, args.strict
    )
    
    # Filter and sort candidates
    candidates = filter_and_sort_candidates(all_metrics, args.topk)
    
    # Save outputs
    Path("results").mkdir(parents=True, exist_ok=True)
    
    all_metrics_path = "results/tech_pairs_all_metrics.csv"
    all_metrics.to_csv(all_metrics_path, index=False)
    log_info(f"Saved all metrics to {all_metrics_path}")
    
    if candidates.empty:
        log_info("WARNING: No candidates found, saving empty candidates CSV")
        candidates.to_csv(args.out_csv, index=False)
    else:
        candidates.to_csv(args.out_csv, index=False)
        log_info(f"Saved {len(candidates)} candidates to {args.out_csv}")
    
    log_info("Done!")


if __name__ == "__main__":
    main()