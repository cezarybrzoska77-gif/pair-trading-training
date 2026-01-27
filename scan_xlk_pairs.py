#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_xlk_pairs.py
-----------------
Skanuje spółki (domyślnie XLK) pod kątem par:
- korelacja (Pearson na stopach zwrotu),
- kointegracja (Engle–Granger, p-value),
- hedge ratio (beta), alpha,
- half-life (szybkość powrotu do średniej).

Zapisuje DWA pliki:
1) xlk_pairs_candidates.csv        -> pary po filtrach (gotowi kandydaci),
2) xlk_pairs_all_metrics.csv       -> wszystkie pary + metryki (do diagnostyki).

Użycie (lokalnie):
    python scan_xlk_pairs.py --start-date 2018-01-01 \
      --out-csv results/xlk_pairs_candidates.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import yfinance as yf


# ======================
# USTAWIENIA DOMYŚLNE
# ======================

DEFAULT_START_DATE = "2018-01-01"
DEFAULT_OUT_CSV = "results/xlk_pairs_candidates.csv"

# Fallback: reprezentanci XLK (możesz rozszerzyć listę)
DEFAULT_XLK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "CRM", "CSCO", "ACN", "ADBE",
    "TXN", "QCOM", "AMD", "INTU", "ORCL", "AMAT", "NOW", "ADI", "MU", "IBM",
    "LRCX", "PANW", "ANSS", "CDNS", "SNPS", "MSI", "FTNT", "MCHP", "KLAC", "NXPI",
    "HPQ", "DELL", "APH", "TEL", "GLW", "FICO", "ROP", "HPE", "CTSH", "PAYX"
]


# ======================
# POMOCNICZE
# ======================

def print_step(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def parse_tickers_from_cli(cli_list: List[str] | None) -> List[str]:
    """Zwraca listę unikalnych tickerów z CLI."""
    if not cli_list:
        return []
    clean = [t.strip().upper() for t in cli_list if t and t.strip()]
    seen = set()
    uniq = []
    for t in clean:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def parse_tickers_from_file(path: str) -> List[str]:
    """Czyta tickery z pliku (jeden ticker na linię)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku z tickerami: {path}")
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#"):
                tickers.append(t)
    return parse_tickers_from_cli(tickers)


def resolve_universe(tickers_cli: List[str], tickers_file: str | None) -> List[str]:
    """Kolejność: --tickers > --tickers-file > fallback XLK."""
    if tickers_cli:
        return tickers_cli
    if tickers_file:
        return parse_tickers_from_file(tickers_file)
    return DEFAULT_XLK_TICKERS.copy()


def extract_prices(
    df: pd.DataFrame,
    field_preferred: str = "Adj Close",
    field_fallback: str = "Close",
    tickers: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Wyciąga macierz cen (wiersze = daty, kolumny = tickery).
    Obsługuje MultiIndex [field, ticker] / [ticker, field] i SingleIndex.
    """
    if df is None or df.empty:
        raise RuntimeError("Brak danych (DataFrame pusty).")

    cols = df.columns

    # MultiIndex: [field, ticker]
    if isinstance(cols, pd.MultiIndex) and field_preferred in cols.get_level_values(0):
        out = df[field_preferred].copy()
        if out.empty and field_fallback in cols.get_level_values(0):
            out = df[field_fallback].copy()
        if out.empty:
            raise KeyError(f"Brak kolumn '{field_preferred}' i '{field_fallback}'.")
        return out

    # MultiIndex: [ticker, field]
    if isinstance(cols, pd.MultiIndex) and (
        field_preferred in cols.get_level_values(-1) or field_fallback in cols.get_level_values(-1)
    ):
        try:
            out = df.xs(field_preferred, axis=1, level=-1)
        except KeyError:
            if field_fallback in cols.get_level_values(-1):
                out = df.xs(field_fallback, axis=1, level=-1)
            else:
                raise KeyError(f"Brak kolumn '{field_preferred}' i '{field_fallback}'.")
        return out

    # SingleIndex
    if field_preferred in cols:
        name = list(tickers)[0] if tickers else field_preferred
        return df[[field_preferred]].rename(columns={field_preferred: name})
    if field_fallback in cols:
        name = list(tickers)[0] if tickers else field_fallback
        return df[[field_fallback]].rename(columns={field_fallback: name})

    raise KeyError(f"Nie znaleziono '{field_preferred}' ani '{field_fallback}'. Kolumny: {list(map(str, cols))}")


def download_prices(
    tickers: List[str],
    start_date: str,
    end_date: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Pobiera dane przez yfinance i zwraca macierz cen (Adj Close preferowane)."""
    print_step(f"Pobieranie danych ({len(tickers)} tickerów) od {start_date}{' do ' + end_date if end_date else ''}…")
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column"
    )
    if data is None or data.empty:
        raise RuntimeError("Brak danych z yfinance. Sprawdź tickery, daty lub sieć.")

    if auto_adjust:
        prices = extract_prices(data, field_preferred="Close", field_fallback="Close", tickers=tickers)
    else:
        prices = extract_prices(data, field_preferred="Adj Close", field_fallback="Close", tickers=tickers)

    # Czyszczenie
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError("Po czyszczeniu brak danych cenowych.")
    min_non_na = max(50, int(0.4 * len(prices)))  # min 50 dni i >=40% dostępnych obserwacji
    prices = prices.dropna(axis=1, thresh=min_non_na).sort_index(axis=1)

    print_step(f"Dane OK: kształt = {prices.shape}")
    return prices


def rolling_corr_last_window(prices: pd.DataFrame, a: str, b: str, lookback: int) -> Tuple[float, int]:
    """Korelacja Pearsona dziennych stóp zwrotu dla ostatniego okna."""
    pa = prices[a].dropna()
    pb = prices[b].dropna()
    common = pa.index.intersection(pb.index)
    if len(common) < 5:
        return (np.nan, 0)
    pa = pa.loc[common].tail(lookback)
    pb = pb.loc[common].tail(lookback)
    ra = pa.pct_change().dropna()
    rb = pb.pct_change().dropna()
    common2 = ra.index.intersection(rb.index)
    if len(common2) < max(10, int(0.6 * lookback)):
        return (np.nan, len(common2))
    corr = ra.loc[common2].corr(rb.loc[common2])
    return (float(corr), len(common2))


@dataclass
class PairMetrics:
    a: str
    b: str
    corr: float
    corr_obs: int
    pvalue: float
    stat: float
    alpha: float
    beta: float
    half_life: float
    sample: int


def compute_hedge_and_spread(y: pd.Series, x: pd.Series) -> Tuple[float, float, pd.Series]:
    """Regresja OLS: y ~ alpha + beta*x -> (alpha, beta, spread)."""
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < 30:
        return (np.nan, np.nan, pd.Series(dtype=float))
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    alpha = float(model.params.get("const", 0.0))
    beta = float(model.params.get("x", np.nan))
    spread = df["y"] - (alpha + beta * df["x"])
    spread.name = "spread"
    return alpha, beta, spread


def compute_half_life(spread: pd.Series) -> float:
    """
    Szacuje half-life (czas powrotu do średniej).
    Model: Δs_t = ρ * s_{t-1} + ε_t,   half_life = -ln(2)/ln(1+ρ)
    """
    s = spread.dropna()
    if len(s) < 40:
        return np.nan
    s_lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    idx = s_lag.index.intersection(ds.index)
    if len(idx) < 30:
        return np.nan
    y = ds.loc[idx].values
    x = s_lag.loc[idx].values
    # OLS bez stałej
    denom = np.dot(x, x)
    if denom == 0:
        return np.nan
    rho = np.dot(x, y) / denom
    if np.isfinite(rho) and rho > -0.999 and (1 + rho) > 0:
        return float(-np.log(2) / np.log(1 + rho))
    return np.nan


def compute_pair_metrics(
    prices: pd.DataFrame,
    a: str,
    b: str,
    corr_lookback: int,
    coint_lookback: int
) -> PairMetrics | None:
    """Liczy metryki pary (corr, pvalue, alpha, beta, half-life)."""
    pa = prices[a].dropna()
    pb = prices[b].dropna()
    common = pa.index.intersection(pb.index)
    if len(common) < max(80, coint_lookback // 2):
        return None

    # Korelacja (ostatnie okno)
    corr, corr_obs = rolling_corr_last_window(prices, a, b, corr_lookback)

    # Pod dane do kointegracji/hedge/half-life (ostatnie coint_lookback dni)
    pa2 = pa.loc[common].tail(coint_lookback)
    pb2 = pb.loc[common].tail(coint_lookback)
    if len(pa2) < 30 or len(pb2) < 30:
        return None

    # Kointegracja (Engle–Granger)
    try:
        stat, pvalue, _ = coint(pa2, pb2)
    except Exception:
        return None

    # Hedge + spread + half-life
    alpha, beta, spread = compute_hedge_and_spread(pa2, pb2)
    hl = compute_half_life(spread) if spread.size > 0 else np.nan

    return PairMetrics(
        a=a, b=b,
        corr=corr if np.isfinite(corr) else np.nan,
        corr_obs=corr_obs,
        pvalue=float(pvalue) if np.isfinite(pvalue) else np.nan,
        stat=float(stat) if np.isfinite(stat) else np.nan,
        alpha=alpha, beta=beta, half_life=hl,
        sample=int(min(len(pa2), len(pb2)))
    )


def scan_pairs(
    prices: pd.DataFrame,
    corr_lookback: int = 90,
    coint_lookback: int = 200,
    corr_min: float = 0.70,
    pvalue_max: float = 0.10,
    half_life_max: float = 30,
    min_sample: int = 150,
    topk: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Liczy metryki dla WSZYSTKICH par i zwraca:
      - df_filtered: po filtrach i sortowaniu,
      - df_all: wszystkie pary (bez filtrów).

    Domyślne progi są celowo łagodne, żeby uniknąć pustych wyników na starcie.
    """
    tickers = list(prices.columns)
    pairs = list(itertools.combinations(tickers, 2))
    results: List[PairMetrics] = []

    print_step(f"Liczenie metryk dla {len(pairs)} par… (to może chwilę potrwać)")

    for idx, (a, b) in enumerate(pairs, start=1):
        pm = compute_pair_metrics(prices, a, b, corr_lookback, coint_lookback)
        if pm is not None:
            results.append(pm)
        if idx % 200 == 0:
            print_step(f"Przetworzono {idx}/{len(pairs)} par…")

    if not results:
        empty = pd.DataFrame(columns=[
            "a", "b", "corr", "corr_obs", "pvalue", "stat", "alpha", "beta", "half_life", "sample"
        ])
        print_step("Brak policzonych metryk (puste wyniki).")
        return empty, empty

    df_all = pd.DataFrame([pm.__dict__ for pm in results])

    # Filtry
    mask = (
        df_all["corr"].ge(corr_min) &
        df_all["pvalue"].le(pvalue_max) &
        df_all["half_life"].le(half_life_max) &
        df_all["sample"].ge(min_sample)
    )
    df_filtered = df_all.loc[mask].copy()

    # Sortowanie jakościowe
    if not df_filtered.empty:
        df_filtered = df_filtered.sort_values(
            by=["pvalue", "half_life", "corr"],
            ascending=[True, True, False]
        )
        if topk is not None and topk > 0:
            df_filtered = df_filtered.head(topk).reset_index(drop=True)

    print_step(f"Par ogółem (metryki policzone): {len(df_all)}")
    print_step(f"Par po filtrach: {len(df_filtered)}  "
               f"[corr≥{corr_min}, p≤{pvalue_max}, hl≤{half_life_max}, sample≥{min_sample}]")

    return df_filtered, df_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Skanner par (XLK) z korelacją i kointegracją.")
    parser.add_argument("--tickers", nargs="*", help="Lista tickerów, np. --tickers MSFT AAPL NVDA")
    parser.add_argument("--tickers-file", default=None, help="Plik z tickerami (jeden na linię).")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help=f"Data początkowa (YYYY-MM-DD), domyślnie {DEFAULT_START_DATE}")
    parser.add_argument("--end-date", default=None, help="Data końcowa (YYYY-MM-DD), opcjonalnie.")
    parser.add_argument("--auto-adjust", action="store_true", help="Jeśli True, Close będzie już skorygowany.")
    parser.add_argument("--corr-lookback", type=int, default=90, help="Okno korelacji (dni), domyślnie 90.")
    parser.add_argument("--coint-lookback", type=int, default=200, help="Okno kointegracji/hedge/half-life, domyślnie 200.")
    parser.add_argument("--corr-min", type=float, default=0.70, help="Min korelacja Pearsona (returns), domyślnie 0.70.")
    parser.add_argument("--pvalue-max", type=float, default=0.10, help="Max p-value (Engle–Granger), domyślnie 0.10.")
    parser.add_argument("--half-life-max", type=float, default=30, help="Max half-life spreadu (dni), domyślnie 30.")
    parser.add_argument("--min-sample", type=int, default=150, help="Min wspólnej historii (dni), domyślnie 150.")
    parser.add_argument("--topk", type=int, default=200, help="Ile najlepszych par zwrócić, domyślnie 200.")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help=f"Ścieżka CSV dla kandydatów, domyślnie {DEFAULT_OUT_CSV}")

    args = parser.parse_args()

    # 1) Wszechświat
    tickers_cli = parse_tickers_from_cli(args.tickers)
    try:
        universe = resolve_universe(tickers_cli, args.tickers_file)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if not universe:
        print("[ERROR] Pusta lista tickerów.", file=sys.stderr)
        return 2

    # 2) Dane
    try:
        prices = download_prices(
            tickers=universe,
            start_date=args.start_date,
            end_date=args.end_date,
            auto_adjust=args.auto_adjust
        )
    except Exception as e:
        print(f"[ERROR] Pobieranie danych nie powiodło się: {e}", file=sys.stderr)
        return 3

    # 3) Skan par
    try:
        df_filtered, df_all = scan_pairs(
            prices=prices,
            corr_lookback=args.corr_lookback,
            coint_lookback=args.coint_lookback,
            corr_min=args.corr_min,
            pvalue_max=args.pvalue_max,
            half_life_max=args.half_life_max,
            min_sample=args.min_sample,
            topk=args.topk
        )
    except Exception as e:
        print(f"[ERROR] Skanowanie par nie powiodło się: {e}", file=sys.stderr)
        return 4

    # 4) Zapis CSV (kandydaci + wszystkie metryki)
    try:
        out_dir = os.path.dirname(args.out_csv) or "."
        os.makedirs(out_dir, exist_ok=True)

        # 4a) kandydaci
        df_filtered.to_csv(args.out_csv, index=False)
        print_step(f"Zapisano {len(df_filtered)} par po filtrach -> {args.out_csv}")

        # 4b) pełne metryki
        out_all = os.path.join(out_dir, "xlk_pairs_all_metrics.csv")
        df_all.to_csv(out_all, index=False)
        print_step(f"Zapisano pełne metryki -> {out_all}")
    except Exception as e:
        print(f"[ERROR] Zapis CSV nie powiódł się: {e}", file=sys.stderr)
        return 5

    # Podgląd (jeśli coś jest)
    if not df_filtered.empty:
        print_step("Top 5 (po filtrach):")
        print(df_filtered.head(5).to_string(index=False))
    else:
        print_step("Po filtrach brak par (sprawdź all_metrics, by zobaczyć wartości i dostroić progi).")

    return 0


if __name__ == "__main__":
    sys.exit(main())