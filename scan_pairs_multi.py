#!/usr/bin/env env python3
"""
scan_pairs_multi.py
===================
Senior quant-grade pairs screening z:
  - wieloma metrykami (Pearson, Spearman, residual corr, EG cointegration, hitrate)
  - klasami A / B+ / B
  - stabilizacją list (histereza ENTRY/STAY/DROP)
  - persistence tracking (first_seen, last_seen, days_since)
  - global scoring (0-1)
  
Usage:
    python scan_pairs_multi.py --universe tech_core
    python scan_pairs_multi.py --universe semis --lookback 365
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import json

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

LOOKBACK_DAYS = 365  # default
MIN_SAMPLE = 200

# Klasyfikacja thresholds
CLASS_A = {
    "corr_mean": 0.82,
    "coint_pvalue": 0.05,
    "hitrate": 0.70,
    "residual_corr": 0.60,
    "sample": 200,
}
CLASS_BP = {  # B+
    "corr_mean": 0.78,
    "corr60_alt": 0.80,
    "corr90_alt": 0.76,
    "coint_pvalue": 0.12,
    "spearman": 0.75,
    "residual_corr": 0.60,
    "hitrate": 0.70,
    "sample": 200,
}
CLASS_B = {  # opcjonalnie wyłączone
    "corr_mean": 0.75,
    "coint_pvalue": 0.15,
    "spearman": 0.70,
    "residual_corr": 0.55,
    "sample": 200,
}

# Stabilizacja: ENTRY / STAY / DROP
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

# Global scoring weights
SCORING_WEIGHTS = {
    "corr_mean": 0.30,
    "spearman_mean": 0.15,
    "residual_corr_max": 0.20,
    "coint_pvalue_best": 0.20,
    "hitrate": 0.10,
    "vol_ratio": 0.05,
}

SCORING_BONUSES = {
    "A": 0.03,
    "B+": 0.01,
    "B": 0.00,
}


# ============================================================================
# DANE: Ściąganie, winsoryzacja, log-returns
# ============================================================================
def download_universe(tickers: List[str], lookback_days: int = 365) -> pd.DataFrame:
    """Pobiera dane OHLC, zwraca Close."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 50)
    
    print(f"[INFO] Downloading {len(tickers)} tickers from {start.date()} to {end.date()}...")
    data = yf.download(tickers, start=start, end=end, progress=False, threads=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]] if "Close" in data.columns else data
    
    close = close.dropna(axis=1, how="all")
    print(f"[INFO] Downloaded {len(close.columns)} valid tickers.")
    return close


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """Winsoryzacja kwantylowa."""
    if s.isna().all():
        return s
    low_val = s.quantile(lower)
    high_val = s.quantile(upper)
    return s.clip(lower=low_val, upper=high_val)


def compute_log_returns(prices: pd.DataFrame, winsorize=True) -> pd.DataFrame:
    """Log-returns z opcjonalną winsoryzacją."""
    log_ret = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    if winsorize:
        log_ret = log_ret.apply(winsorize_series, axis=0)
    return log_ret


# ============================================================================
# METRYKI
# ============================================================================
def compute_pearson_corr(ret1: pd.Series, ret2: pd.Series, window: int) -> float:
    """Rolling Pearson na ostatnich `window` obserwacjach."""
    df = pd.concat([ret1, ret2], axis=1).dropna()
    if len(df) < window:
        return np.nan
    df_tail = df.tail(window)
    if df_tail.std().min() < 1e-9:
        return np.nan
    return df_tail.corr().iloc[0, 1]


def compute_spearman_corr(ret1: pd.Series, ret2: pd.Series, window: int) -> float:
    """Spearman na ostatnich `window` obserwacjach."""
    df = pd.concat([ret1, ret2], axis=1).dropna()
    if len(df) < window:
        return np.nan
    df_tail = df.tail(window)
    try:
        rho, _ = spearmanr(df_tail.iloc[:, 0], df_tail.iloc[:, 1])
        return rho
    except:
        return np.nan


def compute_residual_corr(
    ret1: pd.Series, ret2: pd.Series, ret_index: pd.Series, window: int
) -> float:
    """
    Residual correlation:
      1. Regresja ret1 ~ beta1*ret_index
      2. Regresja ret2 ~ beta2*ret_index
      3. Korelacja residuals
    """
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
    """
    Hitrate: % okien 30-dniowych (w ostatnich 6M), gdzie corr ≥ 0.80.
    """
    df = pd.concat([ret1, ret2], axis=1).dropna()
    # 6M ~ 126 dni handlowych
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


def compute_engle_granger(
    price1: pd.Series, price2: pd.Series, lookbacks: List[int]
) -> Tuple[float, float, int]:
    """
    Kointegracja Engle-Granger dla różnych lookbacks.
    Zwraca: (best_pvalue, best_stat, best_lookback)
    """
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
    """
    Spread beta-neutral: P1 - beta*P2, gdzie beta z regresji P1~P2.
    vol_ratio = std(spread) / |mean(spread)|
    """
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


# ============================================================================
# PERSISTENCE TRACKING
# ============================================================================
def load_persistence(universe: str) -> pd.DataFrame:
    """Wczytuje historię persistence dla danego universe."""
    fpath = PERSISTENCE_DIR / f"{universe}_persistence.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, parse_dates=["first_seen_date", "last_seen_date"])
        return df
    else:
        return pd.DataFrame(
            columns=[
                "pair",
                "first_seen_date",
                "last_seen_date",
                "persistence_days",
                "days_since_last_seen",
                "rolling_class",
                "drop_counter",
            ]
        )


def save_persistence(universe: str, df: pd.DataFrame):
    """Zapisuje persistence."""
    fpath = PERSISTENCE_DIR / f"{universe}_persistence.csv"
    df.to_csv(fpath, index=False)
    print(f"[INFO] Persistence saved: {fpath}")


def update_persistence(
    old_persist: pd.DataFrame,
    current_pairs: pd.DataFrame,
    today: datetime,
) -> pd.DataFrame:
    """
    Aktualizuje persistence tracking:
      - first_seen_date
      - last_seen_date
      - persistence_days
      - days_since_last_seen
      - rolling_class
      - drop_counter (ile dni z rzędu para nie spełnia STAY)
    
    current_pairs musi mieć kolumny: pair, class_assigned
    """
    # Przygotuj dict z old
    persist_dict = {}
    for _, row in old_persist.iterrows():
        persist_dict[row["pair"]] = row.to_dict()
    
    # Przejdź przez current_pairs
    updated_rows = []
    for _, crow in current_pairs.iterrows():
        pair_name = crow["pair"]
        assigned_class = crow.get("class_assigned", "")
        
        if pair_name in persist_dict:
            # Para już istniała
            old_row = persist_dict[pair_name]
            first_seen = old_row["first_seen_date"]
            last_seen = today
            
            # Oblicz persistence_days
            delta = (last_seen - first_seen).days
            persistence_days = delta
            
            # rolling_class
            rolling_class = assigned_class if assigned_class else old_row.get("rolling_class", "")
            
            # drop_counter
            drop_counter = old_row.get("drop_counter", 0)
            
            updated_rows.append(
                {
                    "pair": pair_name,
                    "first_seen_date": first_seen,
                    "last_seen_date": last_seen,
                    "persistence_days": persistence_days,
                    "days_since_last_seen": 0,
                    "rolling_class": rolling_class,
                    "drop_counter": drop_counter,
                }
            )
        else:
            # Nowa para
            updated_rows.append(
                {
                    "pair": pair_name,
                    "first_seen_date": today,
                    "last_seen_date": today,
                    "persistence_days": 0,
                    "days_since_last_seen": 0,
                    "rolling_class": assigned_class if assigned_class else "",
                    "drop_counter": 0,
                }
            )
    
    # Pary, które zniknęły z current → zwiększ days_since_last_seen
    current_set = set(current_pairs["pair"])
    for pair_name, old_row in persist_dict.items():
        if pair_name not in current_set:
            last_seen = pd.to_datetime(old_row["last_seen_date"])
            delta_days = (today - last_seen).days
            updated_rows.append(
                {
                    "pair": pair_name,
                    "first_seen_date": old_row["first_seen_date"],
                    "last_seen_date": old_row["last_seen_date"],
                    "persistence_days": old_row["persistence_days"],
                    "days_since_last_seen": delta_days,
                    "rolling_class": old_row.get("rolling_class", ""),
                    "drop_counter": old_row.get("drop_counter", 0),
                }
            )
    
    df_new = pd.DataFrame(updated_rows)
    return df_new


# ============================================================================
# KLASYFIKACJA & STABILIZACJA
# ============================================================================
def classify_pair(row: pd.Series) -> str:
    """
    Klasyfikuje parę do A / B+ / B / None na podstawie ENTRY thresholds.
    """
    # A-klasa
    if (
        row["corr_mean"] >= CLASS_A["corr_mean"]
        and row["coint_pvalue_best"] <= CLASS_A["coint_pvalue"]
        and row["hitrate_30d_6m"] >= CLASS_A["hitrate"]
        and row["residual_corr_90"] >= CLASS_A["residual_corr"]
        and row["sample"] >= CLASS_A["sample"]
    ):
        return "A"
    
    # B+-klasa
    corr_mean_ok = row["corr_mean"] >= CLASS_BP["corr_mean"]
    corr_alt_ok = (
        row["corr_60"] >= CLASS_BP["corr60_alt"]
        and row["corr_90"] >= CLASS_BP["corr90_alt"]
    )
    if (
        (corr_mean_ok or corr_alt_ok)
        and row["coint_pvalue_best"] <= CLASS_BP["coint_pvalue"]
        and row["spearman_90"] >= CLASS_BP["spearman"]
        and row["residual_corr_90"] >= CLASS_BP["residual_corr"]
        and row["hitrate_30d_6m"] >= CLASS_BP["hitrate"]
        and row["sample"] >= CLASS_BP["sample"]
    ):
        return "B+"
    
    # B-klasa (opcjonalnie wyłączone)
    # if (
    #     row["corr_mean"] >= CLASS_B["corr_mean"]
    #     and row["coint_pvalue_best"] <= CLASS_B["coint_pvalue"]
    #     and row["spearman_90"] >= CLASS_B["spearman"]
    #     and row["residual_corr_90"] >= CLASS_B["residual_corr"]
    #     and row["sample"] >= CLASS_B["sample"]
    # ):
    #     return "B"
    
    return ""


def check_stay(row: pd.Series) -> bool:
    """
    Sprawdza czy para spełnia STAY thresholds (łagodniejsze niż ENTRY).
    """
    if (
        row["corr_mean"] >= STAY_THRESHOLDS["corr_mean"]
        and row["coint_pvalue_best"] <= STAY_THRESHOLDS["coint_pvalue"]
        and row["hitrate_30d_6m"] >= STAY_THRESHOLDS["hitrate"]
        and row["residual_corr_90"] >= STAY_THRESHOLDS["residual_corr"]
    ):
        return True
    return False


def check_drop(row: pd.Series) -> bool:
    """
    Sprawdza czy para spełnia DROP thresholds (bardzo luźne).
    """
    if (
        row["corr_mean"] < DROP_THRESHOLDS["corr_mean"]
        or row["coint_pvalue_best"] > DROP_THRESHOLDS["coint_pvalue"]
        or row["hitrate_30d_6m"] < DROP_THRESHOLDS["hitrate"]
    ):
        return True
    return False


def apply_stability_logic(df_metrics: pd.DataFrame, df_persist: pd.DataFrame) -> pd.DataFrame:
    """
    Stosuje histerezę ENTRY/STAY/DROP:
      1. Klasyfikuj każdą parę (ENTRY)
      2. Jeśli para była wcześniej w klasie A/B+ i teraz spełnia STAY → zostaw
      3. Jeśli para nie spełnia STAY → zwiększ drop_counter
      4. Jeśli drop_counter >= 3 → rolling_class = "drop"
    
    Zwraca: df_metrics z dodatkową kolumną "stable_class"
    """
    persist_dict = df_persist.set_index("pair").to_dict("index")
    
    stable_rows = []
    for _, row in df_metrics.iterrows():
        pair_name = row["pair"]
        entry_class = row["class_assigned"]
        
        # Sprawdź persistence
        if pair_name in persist_dict:
            old_class = persist_dict[pair_name].get("rolling_class", "")
            drop_counter = persist_dict[pair_name].get("drop_counter", 0)
            
            # Jeśli stara klasa A/B+ i spełnia STAY → zostaw
            if old_class in ["A", "B+"] and check_stay(row):
                stable_class = old_class
                new_drop_counter = 0
            # Jeśli nie spełnia STAY → zwiększ drop_counter
            elif check_drop(row):
                new_drop_counter = drop_counter + 1
                if new_drop_counter >= DROP_THRESHOLDS["consecutive_days"]:
                    stable_class = "drop"
                else:
                    stable_class = old_class if old_class else entry_class
            else:
                # Spełnia jakieś warunki, ale nie jest A/B+
                stable_class = entry_class
                new_drop_counter = 0
        else:
            # Nowa para
            stable_class = entry_class
            new_drop_counter = 0
        
        row_dict = row.to_dict()
        row_dict["stable_class"] = stable_class
        row_dict["drop_counter"] = new_drop_counter
        stable_rows.append(row_dict)
    
    df_stable = pd.DataFrame(stable_rows)
    return df_stable


# ============================================================================
# GLOBAL SCORING
# ============================================================================
def compute_global_score(row: pd.Series) -> float:
    """
    Global score (0-1):
      - corr_mean → 30%
      - spearman_mean → 15%
      - residual_corr_max → 20%
      - coint_pvalue_best → 20% (odwrócone: 1 - pvalue)
      - hitrate → 10%
      - vol_ratio → 5% (odwrócone: 1/vol_ratio)
      + bonus za klasę
    """
    score = 0.0
    
    # corr_mean
    score += SCORING_WEIGHTS["corr_mean"] * min(max(row["corr_mean"], 0), 1)
    
    # spearman_mean
    spear_mean = (row["spearman_60"] + row["spearman_90"]) / 2
    score += SCORING_WEIGHTS["spearman_mean"] * min(max(spear_mean, 0), 1)
    
    # residual_corr_max
    resid_max = max(row["residual_corr_60"], row["residual_corr_90"])
    score += SCORING_WEIGHTS["residual_corr_max"] * min(max(resid_max, 0), 1)
    
    # coint_pvalue_best (odwrócone)
    pval = row["coint_pvalue_best"]
    score += SCORING_WEIGHTS["coint_pvalue_best"] * (1 - min(pval, 1))
    
    # hitrate
    score += SCORING_WEIGHTS["hitrate"] * min(max(row["hitrate_30d_6m"], 0), 1)
    
    # vol_ratio (opcjonalnie)
    vol_ratio = row.get("vol_ratio_approx_90", np.nan)
    if pd.notna(vol_ratio) and vol_ratio > 0:
        # Im mniejszy vol_ratio, tym lepiej → 1/vol_ratio, ale cap at 1
        score += SCORING_WEIGHTS["vol_ratio"] * min(1.0 / vol_ratio, 1)
    
    # Bonus za klasę
    stable_class = row.get("stable_class", "")
    if stable_class in SCORING_BONUSES:
        score += SCORING_BONUSES[stable_class]
    
    return min(score, 1.0)


# ============================================================================
# GŁÓWNA FUNKCJA SCREENINGU
# ============================================================================
def screen_pairs(
    universe: str,
    tickers: List[str],
    residual_index: str,
    lookback_days: int = 365,
    min_sample: int = 200,
) -> pd.DataFrame:
    """
    Główna funkcja screeningu par dla danego universe.
    
    Returns:
        DataFrame z wszystkimi metrykami + klasyfikacją + persistence
    """
    today = datetime.now()
    
    # 1. Pobierz dane
    all_tickers = list(set(tickers + [residual_index]))
    prices = download_universe(all_tickers, lookback_days=lookback_days)
    
    # 2. Log-returns
    returns = compute_log_returns(prices, winsorize=True)
    
    # 3. Index returns
    if residual_index not in returns.columns:
        print(f"[WARNING] Residual index {residual_index} not found. Skipping residual corr.")
        ret_index = None
    else:
        ret_index = returns[residual_index]
    
    # 4. Generuj pary
    valid_tickers = [t for t in tickers if t in returns.columns]
    pairs = []
    for i, t1 in enumerate(valid_tickers):
        for t2 in valid_tickers[i + 1 :]:
            pairs.append((t1, t2))
    
    print(f"[INFO] Screening {len(pairs)} pairs in universe '{universe}'...")
    
    # 5. Oblicz metryki dla każdej pary
    results = []
    for t1, t2 in pairs:
        ret1 = returns[t1]
        ret2 = returns[t2]
        price1 = prices[t1]
        price2 = prices[t2]
        
        # Sample size
        df_common = pd.concat([ret1, ret2], axis=1).dropna()
        sample = len(df_common)
        if sample < min_sample:
            continue
        
        # Pearson
        corr_60 = compute_pearson_corr(ret1, ret2, window=60)
        corr_90 = compute_pearson_corr(ret1, ret2, window=90)
        corr_mean = (corr_60 + corr_90) / 2
        
        # Spearman
        spearman_60 = compute_spearman_corr(ret1, ret2, window=60)
        spearman_90 = compute_spearman_corr(ret1, ret2, window=90)
        
        # Residual corr
        if ret_index is not None:
            resid_corr_60 = compute_residual_corr(ret1, ret2, ret_index, window=60)
            resid_corr_90 = compute_residual_corr(ret1, ret2, ret_index, window=90)
        else:
            resid_corr_60 = np.nan
            resid_corr_90 = np.nan
        
        # Hitrate
        hitrate = compute_hitrate_30d_6m(ret1, ret2)
        
        # Engle-Granger
        pval, stat, lb = compute_engle_granger(price1, price2, lookbacks=[240, 300])
        
        # Vol ratio
        vol_ratio = compute_vol_ratio_approx(price1, price2, window=90)
        
        # Obserwacje
        corr_obs_60 = min(len(df_common), 60)
        corr_obs_90 = min(len(df_common), 90)
        
        results.append(
            {
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
            }
        )
    
    df_metrics = pd.DataFrame(results)
    
    if df_metrics.empty:
        print(f"[WARNING] No valid pairs found for {universe}.")
        return df_metrics
    
    # 6. Klasyfikacja ENTRY
    df_metrics["class_assigned"] = df_metrics.apply(classify_pair, axis=1)
    
    # 7. Persistence tracking
    old_persist = load_persistence(universe)
    df_persist = update_persistence(old_persist, df_metrics, today)
    
    # 8. Stabilizacja (histereza)
    df_stable = apply_stability_logic(df_metrics, df_persist)
    
    # 9. Aktualizuj persistence z nowymi wartościami stable_class i drop_counter
    persist_update = []
    for _, row in df_stable.iterrows():
        pair_name = row["pair"]
        stable_class = row["stable_class"]
        drop_counter = row["drop_counter"]
        
        # Znajdź w df_persist
        mask = df_persist["pair"] == pair_name
        if mask.any():
            idx = df_persist[mask].index[0]
            df_persist.at[idx, "rolling_class"] = stable_class
            df_persist.at[idx, "drop_counter"] = drop_counter
    
    save_persistence(universe, df_persist)
    
    # 10. Merge persistence do df_stable
    df_stable = df_stable.merge(
        df_persist[["pair", "first_seen_date", "last_seen_date", "persistence_days", "days_since_last_seen"]],
        on="pair",
        how="left",
    )
    
    # 11. Global scoring
    df_stable["global_score"] = df_stable.apply(compute_global_score, axis=1)
    
    # 12. Sortuj
    df_stable = df_stable.sort_values("global_score", ascending=False).reset_index(drop=True)
    
    print(f"[INFO] Screening complete. Found {len(df_stable)} pairs.")
    return df_stable


# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_results(universe: str, df: pd.DataFrame):
    """Zapisuje wyniki do CSV."""
    univ_dir = RESULTS_DIR / universe
    univ_dir.mkdir(parents=True, exist_ok=True)
    
    # All metrics
    all_file = univ_dir / f"{universe}_all_metrics.csv"
    df.to_csv(all_file, index=False)
    print(f"[INFO] Saved all metrics: {all_file}")
    
    # Candidates (A/B+ z persistence >= 5)
    df_candidates = df[
        (df["stable_class"].isin(["A", "B+"]))
        & (df["persistence_days"] >= 5)
    ].copy()
    cand_file = univ_dir / f"{universe}_candidates.csv"
    df_candidates.to_csv(cand_file, index=False)
    print(f"[INFO] Saved candidates: {cand_file} ({len(df_candidates)} pairs)")
    
    # Stable (wszystkie z stable_class != drop)
    df_stable_list = df[df["stable_class"] != "drop"].copy()
    stable_file = univ_dir / f"{universe}_stable.csv"
    df_stable_list.to_csv(stable_file, index=False)
    print(f"[INFO] Saved stable list: {stable_file} ({len(df_stable_list)} pairs)")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Pairs screening for a single universe")
    parser.add_argument("--universe", type=str, required=True, help="Universe name (e.g. tech_core)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS, help="Lookback days")
    parser.add_argument("--min_sample", type=int, default=MIN_SAMPLE, help="Min sample size")
    args = parser.parse_args()
    
    universe = args.universe
    
    # Load tickers
    ticker_file = Path("data") / f"tickers_{universe}.txt"
    if not ticker_file.exists():
        print(f"[ERROR] Ticker file not found: {ticker_file}")
        return
    
    with open(ticker_file, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # Residual index (pierwsza linia po #RESIDUAL_INDEX:)
    residual_index = None
    tickers = []
    for line in lines:
        if line.startswith("#RESIDUAL_INDEX:"):
            residual_index = line.split(":", 1)[1].strip()
        elif not line.startswith("#"):
            tickers.append(line)
    
    if not residual_index:
        print(f"[ERROR] No #RESIDUAL_INDEX: found in {ticker_file}")
        return
    
    tickers = list(set(tickers))  # Usuń duplikaty
    print(f"[INFO] Universe: {universe}, Residual Index: {residual_index}, Tickers: {len(tickers)}")
    
    # Screen
    df_results = screen_pairs(
        universe=universe,
        tickers=tickers,
        residual_index=residual_index,
        lookback_days=args.lookback,
        min_sample=args.min_sample,
    )
    
    # Save
    if not df_results.empty:
        save_results(universe, df_results)
    else:
        print(f"[WARNING] No results to save for {universe}.")


if __name__ == "__main__":
    main()