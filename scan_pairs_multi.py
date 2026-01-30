#!/usr/bin/env python3
"""
Multi-Universe Pairs Scanner
Retail-friendly pairs trading scanner with cointegration, correlation, and quality gates.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# Universe configurations with default residual indices
UNIVERSE_CONFIG = {
    'tech_core': {'residual_index': 'QQQ', 'file': 'data/tickers_tech_core.txt'},
    'semis': {'residual_index': 'SOXX', 'file': 'data/tickers_semis.txt'},
    'software': {'residual_index': 'IGV', 'file': 'data/tickers_software.txt'},
    'financials': {'residual_index': 'XLF', 'file': 'data/tickers_financials.txt'},
    'healthcare': {'residual_index': 'XLV', 'file': 'data/tickers_healthcare.txt'},
    'discretionary': {'residual_index': 'XLY', 'file': 'data/tickers_discretionary.txt'},
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Scan pairs for a single universe with quality gates'
    )
    
    # Universe selection
    parser.add_argument(
        '--universe',
        choices=list(UNIVERSE_CONFIG.keys()),
        help='Predefined universe name'
    )
    parser.add_argument(
        '--tickers-file',
        type=str,
        help='Custom path to tickers file (one per line)'
    )
    
    # Date range
    parser.add_argument(
        '--start-date',
        type=str,
        default='2018-01-01',
        help='Start date for data download (default: 2018-01-01)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for data download (default: today)'
    )
    
    # Data processing options
    parser.add_argument(
        '--auto-adjust',
        action='store_true',
        default=True,
        help='Use auto-adjusted close prices (default: True)'
    )
    parser.add_argument(
        '--no-auto-adjust',
        action='store_false',
        dest='auto_adjust',
        help='Use raw close prices instead of adjusted'
    )
    parser.add_argument(
        '--use-percent-returns',
        action='store_true',
        default=False,
        help='Use percent returns instead of log returns'
    )
    parser.add_argument(
        '--winsorize',
        action='store_true',
        default=True,
        help='Winsorize returns at 1%/99% (default: True)'
    )
    parser.add_argument(
        '--no-winsorize',
        action='store_false',
        dest='winsorize',
        help='Disable winsorization'
    )
    
    # Residual analysis
    parser.add_argument(
        '--residual-index',
        type=str,
        default=None,
        help='Override residual index ticker (default: auto from universe)'
    )
    
    # Cointegration settings
    parser.add_argument(
        '--coint-lookbacks',
        type=str,
        default='240,300',
        help='Comma-separated cointegration lookback periods (default: 240,300)'
    )
    
    # Filtering
    parser.add_argument(
        '--min-sample',
        type=int,
        default=200,
        help='Minimum sample size for pair (default: 200)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        default=False,
        help='Use only A-grade filters (strict mode)'
    )
    parser.add_argument(
        '--allow-grade-b',
        action='store_true',
        default=False,
        help='Allow B-grade pairs (very loose filters)'
    )
    
    # Output
    parser.add_argument(
        '--topk',
        type=int,
        default=100,
        help='Keep top K pairs per universe (default: 100)'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    args = parser.parse_args()
    
    # Validate universe or tickers-file
    if not args.universe and not args.tickers_file:
        parser.error('Either --universe or --tickers-file must be specified')
    
    return args


def load_tickers(args: argparse.Namespace) -> Tuple[List[str], str]:
    """Load tickers from file and determine universe name."""
    if args.tickers_file:
        ticker_file = Path(args.tickers_file)
        universe_name = ticker_file.stem.replace('tickers_', '')
    else:
        universe_name = args.universe
        ticker_file = Path(UNIVERSE_CONFIG[universe_name]['file'])
    
    if not ticker_file.exists():
        logger.error(f"Ticker file not found: {ticker_file}")
        sys.exit(1)
    
    tickers = []
    with open(ticker_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                tickers.append(line)
    
    # Remove duplicates while preserving order
    tickers = list(dict.fromkeys(tickers))
    
    logger.info(f"Loaded {len(tickers)} unique tickers for universe '{universe_name}'")
    return tickers, universe_name


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    auto_adjust: bool
) -> pd.DataFrame:
    """Download price data using yfinance."""
    logger.info(f"Downloading data from {start_date} to {end_date or 'today'}...")
    
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=auto_adjust,
            progress=False,
            threads=True
        )
        
        if auto_adjust:
            prices = data['Close'] if 'Close' in data else data
        else:
            prices = data['Close'] if 'Close' in data else data
        
        # Handle single ticker case
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        
        # Drop tickers with insufficient data
        min_obs = 100
        valid_tickers = prices.columns[prices.notna().sum() >= min_obs].tolist()
        prices = prices[valid_tickers]
        
        logger.info(f"Downloaded {len(valid_tickers)}/{len(tickers)} tickers with sufficient data")
        logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]} ({len(prices)} days)")
        
        return prices
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        sys.exit(1)


def compute_returns(
    prices: pd.DataFrame,
    use_percent: bool,
    winsorize: bool
) -> pd.DataFrame:
    """Compute returns with optional winsorization."""
    if use_percent:
        logger.info("Computing percent returns...")
        returns = prices.pct_change()
    else:
        logger.info("Computing log returns...")
        returns = np.log(prices / prices.shift(1))
    
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    if winsorize:
        logger.info("Winsorizing returns at 1%/99%...")
        for col in returns.columns:
            lower = returns[col].quantile(0.01)
            upper = returns[col].quantile(0.99)
            returns[col] = returns[col].clip(lower=lower, upper=upper)
    
    return returns


def compute_correlation_metrics(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str
) -> Dict[str, float]:
    """Compute correlation metrics for a pair."""
    ret_a = returns[ticker_a].dropna()
    ret_b = returns[ticker_b].dropna()
    
    # Align series
    common_idx = ret_a.index.intersection(ret_b.index)
    ret_a = ret_a.loc[common_idx]
    ret_b = ret_b.loc[common_idx]
    
    metrics = {}
    
    # Pearson correlations
    for window in [60, 90]:
        min_periods = int(window * 0.6)
        rolling_corr = ret_a.rolling(window, min_periods=min_periods).corr(ret_b)
        
        if len(rolling_corr.dropna()) > 0:
            metrics[f'corr_{window}'] = rolling_corr.iloc[-1]
            metrics[f'corr_obs_{window}'] = rolling_corr.notna().sum()
        else:
            metrics[f'corr_{window}'] = np.nan
            metrics[f'corr_obs_{window}'] = 0
    
    # Spearman correlations
    for window in [60, 90]:
        min_periods = int(window * 0.6)
        if len(ret_a) >= min_periods:
            ret_a_tail = ret_a.iloc[-window:]
            ret_b_tail = ret_b.iloc[-window:]
            if len(ret_a_tail.dropna()) >= min_periods and len(ret_b_tail.dropna()) >= min_periods:
                spearman, _ = stats.spearmanr(ret_a_tail, ret_b_tail, nan_policy='omit')
                metrics[f'spearman_{window}'] = spearman
            else:
                metrics[f'spearman_{window}'] = np.nan
        else:
            metrics[f'spearman_{window}'] = np.nan
    
    # Correlation stability (hit-rate)
    # Last 6 months (~126 trading days), rolling 30-day windows
    if len(ret_a) >= 126:
        ret_a_6m = ret_a.iloc[-126:]
        ret_b_6m = ret_b.iloc[-126:]
        
        hit_count = 0
        total_windows = 0
        
        for i in range(len(ret_a_6m) - 30 + 1):
            window_a = ret_a_6m.iloc[i:i+30]
            window_b = ret_b_6m.iloc[i:i+30]
            
            if len(window_a.dropna()) >= 20 and len(window_b.dropna()) >= 20:
                corr = window_a.corr(window_b)
                if not np.isnan(corr):
                    total_windows += 1
                    if corr >= 0.80:
                        hit_count += 1
        
        if total_windows > 0:
            metrics['corr_hitrate_30d_6m'] = hit_count / total_windows
        else:
            metrics['corr_hitrate_30d_6m'] = np.nan
    else:
        metrics['corr_hitrate_30d_6m'] = np.nan
    
    # Compute mean correlation
    if not np.isnan(metrics.get('corr_60', np.nan)) and not np.isnan(metrics.get('corr_90', np.nan)):
        metrics['corr_mean'] = (metrics['corr_60'] + metrics['corr_90']) / 2
    else:
        metrics['corr_mean'] = np.nan
    
    return metrics


def compute_residual_correlation(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    index_ticker: str
) -> Dict[str, float]:
    """Compute residual correlation after regressing out index."""
    metrics = {}
    
    try:
        # Get aligned prices
        df = prices[[ticker_a, ticker_b, index_ticker]].dropna()
        
        if len(df) < 60:
            return {'resid_corr_60': np.nan, 'resid_corr_90': np.nan}
        
        # Regress each stock on index
        from sklearn.linear_model import LinearRegression
        
        X = df[index_ticker].values.reshape(-1, 1)
        
        model_a = LinearRegression()
        model_a.fit(X, df[ticker_a].values)
        resid_a = df[ticker_a].values - model_a.predict(X)
        
        model_b = LinearRegression()
        model_b.fit(X, df[ticker_b].values)
        resid_b = df[ticker_b].values - model_b.predict(X)
        
        resid_df = pd.DataFrame({
            'resid_a': resid_a,
            'resid_b': resid_b
        }, index=df.index)
        
        # Compute correlation on residuals
        for window in [60, 90]:
            if len(resid_df) >= window:
                tail = resid_df.iloc[-window:]
                corr = tail['resid_a'].corr(tail['resid_b'])
                metrics[f'resid_corr_{window}'] = corr
            else:
                metrics[f'resid_corr_{window}'] = np.nan
        
    except Exception as e:
        logger.warning(f"Error computing residual correlation for {ticker_a}/{ticker_b}: {e}")
        metrics = {'resid_corr_60': np.nan, 'resid_corr_90': np.nan}
    
    return metrics


def compute_cointegration(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookbacks: List[int]
) -> Dict[str, float]:
    """Compute cointegration statistics for multiple lookback periods."""
    df = prices[[ticker_a, ticker_b]].dropna()
    
    best_pvalue = 1.0
    best_stat = 0.0
    best_lookback = 0
    
    for lookback in lookbacks:
        if len(df) >= lookback:
            tail = df.iloc[-lookback:]
            
            try:
                # Engle-Granger cointegration test
                stat, pvalue, _ = coint(tail[ticker_a], tail[ticker_b])
                
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_stat = stat
                    best_lookback = lookback
                    
            except Exception as e:
                logger.debug(f"Cointegration error for {ticker_a}/{ticker_b} at lookback {lookback}: {e}")
                continue
    
    return {
        'coint_stat_best': best_stat,
        'coint_pvalue_best': best_pvalue,
        'coint_lookback_best': best_lookback
    }


def compute_vol_ratio_approx(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str
) -> float:
    """Approximate volatility ratio for the pair (simplified)."""
    try:
        df = prices[[ticker_a, ticker_b]].dropna()
        
        if len(df) < 90:
            return np.nan
        
        tail = df.iloc[-90:]
        
        # Simple hedge-neutral spread approximation: price_a - price_b (normalized)
        spread = tail[ticker_a] - tail[ticker_b]
        
        vol = spread.std()
        mean_level = tail[[ticker_a, ticker_b]].mean().mean()
        
        if mean_level > 0:
            vol_ratio = vol / mean_level
            return vol_ratio
        else:
            return np.nan
            
    except Exception as e:
        logger.debug(f"Vol ratio error for {ticker_a}/{ticker_b}: {e}")
        return np.nan


def scan_pairs(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    residual_index: str,
    coint_lookbacks: List[int],
    min_sample: int
) -> pd.DataFrame:
    """Scan all pairs and compute metrics."""
    tickers = list(prices.columns)
    
    # Remove residual index from pair candidates if present
    if residual_index in tickers:
        tickers = [t for t in tickers if t != residual_index]
    
    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    logger.info(f"Scanning {n_pairs} pairs from {len(tickers)} tickers...")
    
    all_metrics = []
    processed = 0
    
    for i, ticker_a in enumerate(tickers):
        for ticker_b in tickers[i+1:]:
            processed += 1
            
            if processed % 500 == 0:
                logger.info(f"Processed {processed}/{n_pairs} pairs...")
            
            # Check minimum common observations
            common = prices[[ticker_a, ticker_b]].dropna()
            sample_size = len(common)
            
            if sample_size < min_sample:
                continue
            
            # Compute all metrics
            metrics = {
                'a': ticker_a,
                'b': ticker_b,
                'sample': sample_size
            }
            
            # Correlation metrics
            corr_metrics = compute_correlation_metrics(returns, ticker_a, ticker_b)
            metrics.update(corr_metrics)
            
            # Residual correlation
            if residual_index and residual_index in prices.columns:
                resid_metrics = compute_residual_correlation(prices, ticker_a, ticker_b, residual_index)
                metrics.update(resid_metrics)
            else:
                metrics['resid_corr_60'] = np.nan
                metrics['resid_corr_90'] = np.nan
            
            # Cointegration
            coint_metrics = compute_cointegration(prices, ticker_a, ticker_b, coint_lookbacks)
            metrics.update(coint_metrics)
            
            # Volatility ratio (approximate)
            metrics['vol_ratio_approx_90'] = compute_vol_ratio_approx(prices, ticker_a, ticker_b)
            
            all_metrics.append(metrics)
    
    logger.info(f"Completed scanning {processed} pairs, {len(all_metrics)} with sufficient data")
    
    if not all_metrics:
        logger.warning("No pairs found with sufficient data!")
        return pd.DataFrame()
    
    return pd.DataFrame(all_metrics)


def apply_filters(df: pd.DataFrame, strict: bool, allow_grade_b: bool) -> pd.DataFrame:
    """Apply quality filters and assign grades."""
    if df.empty:
        return df
    
    df = df.copy()
    df['grade'] = None
    
    # A-grade filters (priority)
    a_mask = (
        (df['corr_mean'] >= 0.82) &
        (df['coint_pvalue_best'] <= 0.05) &
        (df['sample'] >= 200)
    )
    df.loc[a_mask, 'grade'] = 'A'
    
    if not strict:
        # B+ grade filters (safely loosened)
        b_plus_mask = (
            (
                (df['corr_mean'] >= 0.78) |
                ((df['corr_60'] >= 0.80) & (df['corr_90'] >= 0.76))
            ) &
            (df['coint_pvalue_best'] <= 0.12) &
            (df['spearman_60'] >= 0.75) &
            (df['spearman_90'] >= 0.75) &
            (df[['resid_corr_60', 'resid_corr_90']].max(axis=1) >= 0.60) &
            (df['corr_hitrate_30d_6m'] >= 0.70) &
            (df['sample'] >= 200) &
            (df['grade'].isna())
        )
        df.loc[b_plus_mask, 'grade'] = 'B+'
    
    if allow_grade_b and not strict:
        # B-grade filters (very loose, optional)
        b_mask = (
            (df['corr_mean'] >= 0.75) &
            (df['coint_pvalue_best'] <= 0.15) &
            (df['spearman_60'] >= 0.70) &
            (df['spearman_90'] >= 0.70) &
            (df[['resid_corr_60', 'resid_corr_90']].max(axis=1) >= 0.55) &
            (df['sample'] >= 200) &
            (df['grade'].isna())
        )
        df.loc[b_mask, 'grade'] = 'B'
    
    # Filter to candidates with assigned grade
    candidates = df[df['grade'].notna()].copy()
    
    logger.info(f"Filtering results: {len(candidates)}/{len(df)} pairs passed filters")
    if len(candidates) > 0:
        grade_counts = candidates['grade'].value_counts()
        for grade, count in grade_counts.items():
            logger.info(f"  {grade}-grade: {count} pairs")
    
    return candidates


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted scores for ranking."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Helper function to normalize
    def norm_score(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
        return ((series - min_val) / (max_val - min_val)).clip(0, 1).fillna(0)
    
    # Component scores
    df['score_corr'] = norm_score(df['corr_mean'], 0.60, 0.90)
    
    # Spearman mean
    df['spearman_mean'] = df[['spearman_60', 'spearman_90']].mean(axis=1)
    df['score_spear'] = norm_score(df['spearman_mean'], 0.60, 0.90)
    
    # Residual correlation max
    df['resid_corr_max'] = df[['resid_corr_60', 'resid_corr_90']].max(axis=1)
    df['score_resid'] = norm_score(df['resid_corr_max'], 0.40, 0.80)
    
    # Cointegration (lower p-value is better)
    df['score_coint'] = (1 - (df['coint_pvalue_best'] / 0.15).clip(0, 1)).fillna(0)
    
    # Hit-rate already in [0,1]
    df['score_hitrate'] = df['corr_hitrate_30d_6m'].fillna(0)
    
    # Volatility ratio (optional, simplified)
    if 'vol_ratio_approx_90' in df.columns:
        df['score_vol'] = (1 - (df['vol_ratio_approx_90'] / 6).clip(0, 1)).fillna(0)
    else:
        df['score_vol'] = 0
    
    # Weighted score
    df['score_w'] = (
        0.30 * df['score_corr'] +
        0.15 * df['score_spear'] +
        0.20 * df['score_resid'] +
        0.20 * df['score_coint'] +
        0.10 * df['score_hitrate'] +
        0.05 * df['score_vol']
    )
    
    # Grade bonus
    grade_bonus = {'A': 0.03, 'B+': 0.01, 'B': 0.00}
    df['score_w'] += df['grade'].map(grade_bonus).fillna(0)
    
    # Clip to [0, 1]
    df['score_w'] = df['score_w'].clip(0, 1)
    
    logger.info(f"Computed scores for {len(df)} pairs")
    logger.info(f"Score range: {df['score_w'].min():.3f} - {df['score_w'].max():.3f}")
    
    return df


def save_results(
    all_metrics: pd.DataFrame,
    candidates: pd.DataFrame,
    universe_name: str,
    out_dir: str,
    topk: int
) -> None:
    """Save results to CSV files."""
    universe_dir = Path(out_dir) / universe_name
    universe_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all metrics (unfiltered)
    all_file = universe_dir / f"{universe_name}_all_metrics.csv"
    all_metrics.to_csv(all_file, index=False)
    logger.info(f"Saved all metrics to: {all_file}")
    
    # Save candidates (filtered and scored)
    if not candidates.empty:
        # Sort by score and take top K
        candidates_sorted = candidates.sort_values('score_w', ascending=False).head(topk)
        
        # Add universe column
        candidates_sorted.insert(0, 'universe', universe_name)
        
        candidates_file = universe_dir / f"{universe_name}_candidates.csv"
        candidates_sorted.to_csv(candidates_file, index=False)
        logger.info(f"Saved {len(candidates_sorted)} candidates to: {candidates_file}")
    else:
        logger.warning("No candidates to save!")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load tickers
    tickers, universe_name = load_tickers(args)
    
    if not tickers:
        logger.error("No tickers loaded!")
        sys.exit(1)
    
    # Determine residual index
    if args.residual_index:
        residual_index = args.residual_index
    elif args.universe:
        residual_index = UNIVERSE_CONFIG[args.universe]['residual_index']
    else:
        residual_index = None
        logger.warning("No residual index specified, skipping residual correlation")
    
    # Add residual index to download list if needed
    download_tickers = tickers.copy()
    if residual_index and residual_index not in download_tickers:
        download_tickers.append(residual_index)
    
    # Download data
    prices = download_data(download_tickers, args.start_date, args.end_date, args.auto_adjust)
    
    if prices.empty:
        logger.error("No price data downloaded!")
        sys.exit(1)
    
    # Compute returns
    returns = compute_returns(prices, args.use_percent_returns, args.winsorize)
    
    # Parse cointegration lookbacks
    coint_lookbacks = [int(x.strip()) for x in args.coint_lookbacks.split(',')]
    logger.info(f"Using cointegration lookbacks: {coint_lookbacks}")
    
    # Scan pairs
    all_metrics = scan_pairs(prices, returns, residual_index, coint_lookbacks, args.min_sample)
    
    if all_metrics.empty:
        logger.warning("No pairs found!")
        # Save empty results
        save_results(all_metrics, pd.DataFrame(), universe_name, args.out_dir, args.topk)
        return
    
    # Apply filters
    candidates = apply_filters(all_metrics, args.strict, args.allow_grade_b)
    
    # Compute scores
    if not candidates.empty:
        candidates = compute_scores(candidates)
    
    # Save results
    save_results(all_metrics, candidates, universe_name, args.out_dir, args.topk)
    
    logger.info("=" * 60)
    logger.info(f"SCAN COMPLETE for universe '{universe_name}'")
    logger.info(f"Total pairs analyzed: {len(all_metrics)}")
    logger.info(f"Candidates found: {len(candidates)}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()