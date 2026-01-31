#!/usr/bin/env python3
"""
scan_pairs_multi.py
===================
Pairs screening z error handling i verbose logging.
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr, pearsonr
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
RESULTS_DIR = Path("results")
PERSISTENCE_DIR = RESULTS_DIR / "persistence"
PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_DAYS = 365
MIN_SAMPLE = 200

# Klasyfikacja
CLASS_A = {
    "corr_mean": 0.82,
    "coint_pvalue": 0.05,
    "hitrate": 0.70,
    "residual_corr": 0.60,
    "sample": 200,
}
CLASS_BP = {
    "corr_mean": 0.78,
    "corr60_alt": 0.80,
    "corr90_alt": 0.76,
    "coint_pvalue": 0.12,
    "spearman": 0.75,
    "residual_corr": 0.60,
    "hitrate": 0.70,
    "sample": 200,
}

STAY_THRESHOLDS = {
    "corr_mean": 0.78,
    "coint_pvalue": 0.14,
    "hitrate": 0.65,
    "residual_corr": 0.55,
}
DROP_THRESHOLDS = {
    "corr_mean": 0.75,
    "coint_pvalue": 0.18,
    "hitrate": 0.50,
    "consecutive_days": 3,
}

SCORING_WEIGHTS = {
    "corr_mean": 0.30,
    "spearman_mean": 0.15,
    "residual_corr_max": 0.20,
    "coint_pvalue_best": 0.20,
    "hitrate": 0.10,
    "vol_ratio": 0.05,
}

SCORING_BONUSES = {"A": 0.03, "B+": 0.01, "B": 0.00}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def log(msg: str):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def download_universe(tickers: List[str], lookback_days: int = 365) -> pd.DataFrame:
    """Download OHLC data."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 50)
    
    log(f"Downloading {len(tickers)} tickers from {start.date()} to {end.date()}...")
    
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, threads=True)
    except Exception as e:
        log(f"ERROR downloading data: {e}")
        return pd.DataFrame()
    
    if data.empty:
        log("ERROR: Downloaded data is empty")
        return pd.DataFrame()
    
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        else:
            log("ERROR: No 'Close' column found")
            return pd.DataFrame()
    else:
        close = data[["Close"]] if "Close" in data.columns else data
    
    close = close.dropna(axis=1, how="all")
    log(f"Downloaded {len(close.columns)} valid tickers with {len(close)} rows")
    return close


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    if s.isna().all():
        return s
    low_val = s.quantile(lower)
    high_val = s.quantile(upper)
    return s.clip(lower=low_val, upper=high_val)


def compute_log_returns(prices: pd.DataFrame, winsorize=True) -> pd.DataFrame:
    log_ret = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    if winsorize:
        log_ret = log_ret.apply(winsorize_series, axis=0)
    return log_ret


def compute_pearson_corr(ret1: pd.Series, ret2: pd.Series, window: int) -> float:
    df = pd.concat([ret1, ret2], axis=1).dropna()
    if len(df) < window:
        return np.nan
    df_tail = df.tail(window)
    if df_tail.std().min() < 1e-9:
        return np.nan
    return df_tail.corr().iloc[0, 1]


def compute_spearman_corr(ret1: pd.Series, ret2: pd.Series, window: int) -> float:
    df = pd.concat([ret1, ret2], axis=1).dropna()
    if len(df) < window:
        return np.nan
    df_tail = df.tail(window)
    try:
        rho, _ = spearmanr(df_tail.iloc[:, 0], df_tail.iloc[:, 1])
        return rho
    except:
        return np.nan


def compute_residual_corr(ret1: pd.Series, ret2: pd.Series, ret_index: pd.Series, window: int) -> float:
    df = pd.concat([ret1, ret2, ret_index], axis=1).dropna()
    if len(df) < window:
        return np.nan
    df_tail = df.tail(window)
    X = df_tail.iloc[:, 2].values.reshape(-1, 1)
    y1 = df_tail.iloc[:, 0].values
    y2 = df_tail.iloc[:, 1].values
    
    try:
        model1 = LinearRegression().fit(X, y1)
        resid1 = y1 - model1.predict(X)
        model2 = LinearRegression().fit(X, y2)
        resid2 = y2 - model2.predict(X)
        
        if resid1.std() < 1e-9 or resid2.std() < 1e-9:
            return np.nan
        corr_val, _ = pearsonr(resid1, resid2)
        return corr_val
    except:
        return np.nan


def compute_hitrate_30d_6m(ret1: pd.Series, ret2: pd.Series) -> float:
    df = pd.concat([ret1, ret2], axis=1).dropna()
    df_6m = df.tail(126)
    if len(df_6m) < 30:
        return np.nan
    
    window_size = 30
    hits = 0
    total = 0
    for i in range(len(df_6m) - window_size + 1):
        window = df_6m.iloc[i : i + window_size]
        if window.std().min() < 1e-9:
            continue
        c = window.corr().iloc[0, 1]
        total += 1
        if c >= 0.80:
            hits += 1
    
    return hits / total if total > 0 else np.nan


def compute_engle_granger(price1: pd.Series, price2: pd.Series, lookbacks: List[int]) -> Tuple[float, float, int]:
    df = pd.concat([price1, price2], axis=1).dropna()
    
    best_pval = 1.0
    best_stat = 0.0
    best_lb = lookbacks[0]
    
    for lb in lookbacks:
        if len(df) < lb:
            continue
        df_tail = df.tail(lb)
        try:
            stat, pval, _ = coint(df_tail.iloc[:, 0], df_tail.iloc[:, 1])
            if pval < best_pval:
                best_pval = pval
                best_stat = stat
                best_lb = lb
        except:
            pass
    
    return best_pval, best_stat, best_lb


def compute_vol_ratio_approx(price1: pd.Series, price2: pd.Series, window: int = 90) -> float:
    df = pd.concat([price1, price2], axis=1).dropna()
    if len(df) < window:
        return np.nan
    df_tail = df.tail(window)
    
    X = df_tail.iloc[:, 1].values.reshape(-1, 1)
    y = df_tail.iloc[:, 0].values
    
    try:
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        spread = y - beta * X.flatten()
        mean_spread = spread.mean()
        std_spread = spread.std()
        if abs(mean_spread) < 1e-9:
            return np.nan
        return std_spread / abs(mean_spread)
    except:
        return np.nan


def load_persistence(universe: str) -> pd.DataFrame:
    fpath = PERSISTENCE_DIR / f"{universe}_persistence.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, parse_dates=["first_seen_date", "last_seen_date"])
        return df
    else:
        return pd.DataFrame(
            columns=[
                "pair", "first_seen_date", "last_seen_date",
                "persistence_days", "days_since_last_seen",
                "rolling_class", "drop_counter",
            ]
        )


def save_persistence(universe: str, df: pd.DataFrame):
    fpath = PERSISTENCE_DIR / f"{universe}_persistence.csv"
    df.to_csv(fpath, index=False)
    log(f"Persistence saved: {fpath}")


def update_persistence(old_persist: pd.DataFrame, current_pairs: pd.DataFrame, today: datetime) -> pd.DataFrame:
    persist_dict = {}
    for _, row in old_persist.iterrows():
        persist_dict[row["pair"]] = row.to_dict()
    
    updated_rows = []
    for _, crow in current_pairs.iterrows():
        pair_name = crow["pair"]
        assigned_class = crow.get("class_assigned", "")
        
        if pair_name in persist_dict:
            old_row = persist_dict[pair_name]
            first_seen = old_row["first_seen_date"]
            last_seen = today
            delta = (last_seen - first_seen).days
            persistence_days = delta
            rolling_class = assigned_class if assigned_class else old_row.get("rolling_class", "")
            drop_counter = old_row.get("drop_counter", 0)
            
            updated_rows.append({
                "pair": pair_name,
                "first_seen_date": first_seen,
                "last_seen_date": last_seen,
                "persistence_days": persistence_days,
                "days_since_last_seen": 0,
                "rolling_class": rolling_class,
                "drop_counter": drop_counter,
            })
        else:
            updated_rows.append({
                "pair": pair_name,
                "first_seen_date": today,
                "last_seen_date": today,
                "persistence_days": 0,
                "days_since_last_seen": 0,
                "rolling_class": assigned_class if assigned_class else "",
                "drop_counter": 0,
            })
    
    current_set = set(current_pairs["pair"])
    for pair_name, old_row in persist_dict.items():
        if pair_name not in current_set:
            last_seen = pd.to_datetime(old_row["last_seen_date"])
            delta_days = (today - last_seen).days
            updated_rows.append({
                "pair": pair_name,
                "first_seen_date": old_row["first_seen_date"],
                "last_seen_date": old_row["last_seen_date"],
                "persistence_days": old_row["persistence_days"],
                "days_since_last_seen": delta_days,
                "rolling_class": old_row.get("rolling_class", ""),
                "drop_counter": old_row.get("drop_counter", 0),
            })
    
    df_new = pd.DataFrame(updated_rows)
    return df_new


def classify_pair(row: pd.Series) -> str:
    if (
        row["corr_mean"] >= CLASS_A["corr_mean"]
        and row["coint_pvalue_best"] <= CLASS_A["coint_pvalue"]
        and row["hitrate_30d_6m"] >= CLASS_A["hitrate"]
        and row["residual_corr_90"] >= CLASS_A["residual_corr"]
        and row["sample"] >= CLASS_A["sample"]
    ):
        return "A"
    
    corr_mean_ok = row["corr_mean"] >= CLASS_BP["corr_mean"]
    corr_alt_ok = (row["corr_60"] >= CLASS_BP["corr60_alt"] and row["corr_90"] >= CLASS_BP["corr90_alt"])
    
    if (
        (corr_mean_ok or corr_alt_ok)
        and row["coint_pvalue_best"] <= CLASS_BP["coint_pvalue"]
        and row["spearman_90"] >= CLASS_BP["spearman"]
        and row["residual_corr_90"] >= CLASS_BP["residual_corr"]
        and row["hitrate_30d_6m"] >= CLASS_BP["hitrate"]
        and row["sample"] >= CLASS_BP["sample"]
    ):
        return "B+"
    
    return ""


def check_stay(row: pd.Series) -> bool:
    return (
        row["corr_mean"] >= STAY_THRESHOLDS["corr_mean"]
        and row["coint_pvalue_best"] <= STAY_THRESHOLDS["coint_pvalue"]
        and row["hitrate_30d_6m"] >= STAY_THRESHOLDS["hitrate"]
        and row["residual_corr_90"] >= STAY_THRESHOLDS["residual_corr"]
    )


def check_drop(row: pd.Series) -> bool:
    return (
        row["corr_mean"] < DROP_THRESHOLDS["corr_mean"]
        or row["coint_pvalue_best"] > DROP_THRESHOLDS["coint_pvalue"]
        or row["hitrate_30d_6m"] < DROP_THRESHOLDS["hitrate"]
    )


def apply_stability_logic(df_metrics: pd.DataFrame, df_persist: pd.DataFrame) -> pd.DataFrame:
    persist_dict = df_persist.set_index("pair").to_dict("index")
    
    stable_rows = []
    for _, row in df_metrics.iterrows():
        pair_name = row["pair"]
        entry_class = row["class_assigned"]
        
        if pair_name in persist_dict:
            old_class = persist_dict[pair_name].get("rolling_class", "")
            drop_counter = persist_dict[pair_name].get("drop_counter", 0)
            
            if old_class in ["A", "B+"] and check_stay(row):
                stable_class = old_class
                new_drop_counter = 0
            elif check_drop(row):
                new_drop_counter = drop_counter + 1
                if new_drop_counter >= DROP_THRESHOLDS["consecutive_days"]:
                    stable_class = "drop"
                else:
                    stable_class = old_class if old_class else entry_class
            else:
                stable_class = entry_class
                new_drop_counter = 0
        else:
            stable_class = entry_class
            new_drop_counter = 0
        
        row_dict = row.to_dict()
        row_dict["stable_class"] = stable_class
        row_dict["drop_counter"] = new_drop_counter
        stable_rows.append(row_dict)
    
    df_stable = pd.DataFrame(stable_rows)
    return df_stable


def compute_global_score(row: pd.Series) -> float:
    score = 0.0
    score += SCORING_WEIGHTS["corr_mean"] * min(max(row["corr_mean"], 0), 1)
    spear_mean = (row["spearman_60"] + row["spearman_90"]) / 2
    score += SCORING_WEIGHTS["spearman_mean"] * min(max(spear_mean, 0), 1)
    resid_max = max(row["residual_corr_60"], row["residual_corr_90"])
    score += SCORING_WEIGHTS["residual_corr_max"] * min(max(resid_max, 0), 1)
    pval = row["coint_pvalue_best"]
    score += SCORING_WEIGHTS["coint_pvalue_best"] * (1 - min(pval, 1))
    score += SCORING_WEIGHTS["hitrate"] * min(max(row["hitrate_30d_6m"], 0), 1)
    vol_ratio = row.get("vol_ratio_approx_90", np.nan)
    if pd.notna(vol_ratio) and vol_ratio > 0:
        score += SCORING_WEIGHTS["vol_ratio"] * min(1.0 / vol_ratio, 1)
    stable_class = row.get("stable_class", "")
    if stable_class in SCORING_BONUSES:
        score += SCORING_BONUSES[stable_class]
    return min(score, 1.0)


def screen_pairs(universe: str, tickers: List[str], residual_index: str, lookback_days: int = 365, min_sample: int = 200) -> pd.DataFrame:
    today = datetime.now()
    
    all_tickers = list(set(tickers + [residual_index]))
    prices = download_universe(all_tickers, lookback_days=lookback_days)
    
    if prices.empty:
        log("ERROR: No price data available")
        return pd.DataFrame()
    
    returns = compute_log_returns(prices, winsorize=True)
    
    if residual_index not in returns.columns:
        log(f"WARNING: Residual index {residual_index} not found. Skipping residual corr.")
        ret_index = None
    else:
        ret_index = returns[residual_index]
    
    valid_tickers = [t for t in tickers if t in returns.columns]
    log(f"Valid tickers: {len(valid_tickers)}")
    
    pairs = []
    for i, t1 in enumerate(valid_tickers):
        for t2 in valid_tickers[i + 1 :]:
            pairs.append((t1, t2))
    
    log(f"Screening {len(pairs)} pairs...")
    
    results = []
    for idx, (t1, t2) in enumerate(pairs):
        if idx % 100 == 0 and idx > 0:
            log(f"Processed {idx}/{len(pairs)} pairs...")
        
        ret1 = returns[t1]
        ret2 = returns[t2]
        price1 = prices[t1]
        price2 = prices[t2]
        
        df_common = pd.concat([ret1, ret2], axis=1).dropna()
        sample = len(df_common)
        if sample < min_sample:
            continue
        
        corr_60 = compute_pearson_corr(ret1, ret2, window=60)
        corr_90 = compute_pearson_corr(ret1, ret2, window=90)
        corr_mean = (corr_60 + corr_90) / 2
        
        spearman_60 = compute_spearman_corr(ret1, ret2, window=60)
        spearman_90 = compute_spearman_corr(ret1, ret2, window=90)
        
        if ret_index is not None:
            resid_corr_60 = compute_residual_corr(ret1, ret2, ret_index, window=60)
            resid_corr_90 = compute_residual_corr(ret1, ret2, ret_index, window=90)
        else:
            resid_corr_60 = np.nan
            resid_corr_90 = np.nan
        
        hitrate = compute_hitrate_30d_6m(ret1, ret2)
        pval, stat, lb = compute_engle_granger(price1, price2, lookbacks=[240, 300])
        vol_ratio = compute_vol_ratio_approx(price1, price2, window=90)
        
        corr_obs_60 = min(len(df_common), 60)
        corr_obs_90 = min(len(df_common), 90)
        
        results.append({
            "pair": f"{t1}_{t2}",
            "ticker1": t1,
            "ticker2": t2,
            "sample": sample,
            "corr_60": corr_60,
            "corr_90": corr_90,
            "corr_mean": corr_mean,
            "spearman_60": spearman_60,
            "spearman_90": spearman_90,
            "residual_corr_60": resid_corr_60,
            "residual_corr_90": resid_corr_90,
            "hitrate_30d_6m": hitrate,
            "coint_pvalue_best": pval,
            "coint_stat_best": stat,
            "coint_lookback_best": lb,
            "vol_ratio_approx_90": vol_ratio,
            "corr_obs_60": corr_obs_60,
            "corr_obs_90": corr_obs_90,
        })
    
    df_metrics = pd.DataFrame(results)
    
    if df_metrics.empty:
        log(f"WARNING: No valid pairs found")
        return df_metrics
    
    df_metrics["class_assigned"] = df_metrics.apply(classify_pair, axis=1)
    
    old_persist = load_persistence(universe)
    df_persist = update_persistence(old_persist, df_metrics, today)
    df_stable = apply_stability_logic(df_metrics, df_persist)
    
    for _, row in df_stable.iterrows():
        pair_name = row["pair"]
        stable_class = row["stable_class"]
        drop_counter = row["drop_counter"]
        mask = df_persist["pair"] == pair_name
        if mask.any():
            idx = df_persist[mask].index[0]
            df_persist.at[idx, "rolling_class"] = stable_class
            df_persist.at[idx, "drop_counter"] = drop_counter
    
    save_persistence(universe, df_persist)
    
    df_stable = df_stable.merge(
        df_persist[["pair", "first_seen_date", "last_seen_date", "persistence_days", "days_since_last_seen"]],
        on="pair",
        how="left",
    )
    
    df_stable["global_score"] = df_stable.apply(compute_global_score, axis=1)
    df_stable = df_stable.sort_values("global_score", ascending=False).reset_index(drop=True)
    
    log(f"Screening complete. Found {len(df_stable)} pairs")
    return df_stable


def save_results(universe: str, df: pd.DataFrame):
    univ_dir = RESULTS_DIR / universe
    univ_dir.mkdir(parents=True, exist_ok=True)
    
    all_file = univ_dir / f"{universe}_all_metrics.csv"
    df.to_csv(all_file, index=False)
    log(f"Saved all metrics: {all_file}")
    
    df_candidates = df[(df["stable_class"].isin(["A", "B+"])) & (df["persistence_days"] >= 5)].copy()
    cand_file = univ_dir / f"{universe}_candidates.csv"
    df_candidates.to_csv(cand_file, index=False)
    log(f"Saved candidates: {cand_file} ({len(df_candidates)} pairs)")
    
    df_stable_list = df[df["stable_class"] != "drop"].copy()
    stable_file = univ_dir / f"{universe}_stable.csv"
    df_stable_list.to_csv(stable_file, index=False)
    log(f"Saved stable list: {stable_file} ({len(df_stable_list)} pairs)")


def main():
    parser = argparse.ArgumentParser(description="Pairs screening for a single universe")
    parser.add_argument("--universe", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--min_sample", type=int, default=MIN_SAMPLE)
    args = parser.parse_args()
    
    universe = args.universe
    ticker_file = Path("data") / f"tickers_{universe}.txt"
    
    if not ticker_file.exists():
        log(f"ERROR: Ticker file not found: {ticker_file}")
        sys.exit(1)
    
    with open(ticker_file, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    residual_index = None
    tickers = []
    for line in lines:
        if line.startswith("#RESIDUAL_INDEX:"):
            residual_index = line.split(":", 1)[1].strip()
        elif not line.startswith("#"):
            tickers.append(line)
    
    if not residual_index:
        log(f"ERROR: No #RESIDUAL_INDEX: found in {ticker_file}")
        sys.exit(1)
    
    tickers = list(set(tickers))
    log(f"Universe: {universe}, Residual Index: {residual_index}, Tickers: {len(tickers)}")
    
    df_results = screen_pairs(
        universe=universe,
        tickers=tickers,
        residual_index=residual_index,
        lookback_days=args.lookback,
        min_sample=args.min_sample,
    )
    
    if not df_results.empty:
        save_results(universe, df_results)
        log("SUCCESS: Screening completed")
    else:
        log("WARNING: No results to save")
        sys.exit(0)


if __name__ == "__main__":
    main()