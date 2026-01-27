#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_xlk_pairs.py
-----------------
Skanuje wszechświat spółek (domyślnie XLK) pod kątem par z wysoką korelacją
i kointegracją (Engle–Granger). Zwraca CSV z metrykami (corr, p-value, half-life, beta, alpha).

Użycie (przykłady):
    # 1) Domyślne XLK (fallback lista), od 2018-01-01, filtry: corr>=0.85, p<=0.03, half-life<=15
    python scan_xlk_pairs.py --start-date 2018-01-01

    # 2) Własne tickery
    python scan_xlk_pairs.py --tickers MSFT AAPL NVDA AVGO --start-date 2018-01-01

    # 3) Lista z pliku (jeden ticker na linię)
    python scan_xlk_pairs.py --tickers-file data/xlk_tickers.txt --start-date 2018-01-01

    # 4) Dostosowanie progów i okien
    python scan_xlk_pairs.py --start-date 2018-01-01 --corr-lookback 90 --coint-lookback 250 \
                             --corr-min 0.85 --pvalue-max 0.03 --half-life-max 15 \
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
# KONFIGURACJA DOMYŚLNA
# ======================

DEFAULT_START_DATE = "2018-01-01"
DEFAULT_OUT_CSV = "xlk_pairs_candidates.csv"

# Fallback: płynni reprezentanci XLK (możesz rozszerzyć listę; to tylko start)
DEFAULT_XLK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "CRM", "CSCO", "ACN", "ADBE",
    "TXN", "QCOM", "AMD", "INTU", "ORCL", "AMAT", "NOW", "ADI", "MU", "IBM",
    "LRCX", "PANW", "ANSS", "CDNS", "SNPS", "MSI", "FTNT", "MCHP", "KLAC", "NXPI",
    "HPQ", "DELL", "APH", "TEL", "GLW", "FICO", "ROP", "HPE", "CTSH", "PAYX"
]


# ======================
# NARZĘDZIA
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
    """Czyta tickery z pliku (jeden na linię)."""
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
    """Kolejność priorytetów: --tickers > --tickers-file > fallback XLK."""
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
    Wyciąga macierz cen (wiersze = daty, kolumny = tickery) dla zadanego pola.
    Obsługuje MultiIndex [field, ticker] / [ticker, field] i SingleIndex.
    """
    if df is None or df.empty:
        raise RuntimeError("Brak danych wejściowych (DataFrame jest pusty).")

    cols = df.columns

    # MultiIndex: [field, ticker]
    if isinstance(cols, pd.MultiIndex) and field_preferred in cols.get_level_values(0):
        out = df[field_preferred].copy()
        if out.empty and field_fallback in cols.get_level_values(0):
            out = df[field_fallback].copy()
        if out.empty:
            raise KeyError(f"Nie znaleziono kolumn '{field_preferred}' ani '{field_fallback}'.")
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
                raise KeyError(f"Nie znaleziono kolumn '{field_preferred}' ani '{field_fallback}'.")
        return out

    # SingleIndex:
    if field_preferred in cols:
        name = list(tickers)[0] if tickers else field_preferred
        return df[[field_preferred]].rename(columns={field_preferred: name})
    if field_fallback in cols:
        name = list(tickers)[0] if tickers else field_fallback
        return df[[field_fallback]].rename(columns={field_fallback: name})

    raise KeyError(
        f"Nie znaleziono kolumn '{field_preferred}' ani '{field_fallback}'. "
        f"Dostępne kolumny: {list(map(str, cols))}"
    )


def download_prices(
    tickers: List[str],
    start_date: str,
    end_date: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Pobiera dane via yfinance i zwraca macierz cen (Adj Close preferowane).
    """
    print_step(f"Pobieranie danych ({len(tickers)} tickerów) od {start_date}{' do ' + end_date if end_date else ''}...")
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column"
    )
    if data is None or data.empty:
        raise RuntimeError("Brak danych z yfinance. Sprawdź tickery, daty lub połączenie z siecią.")

    if auto_adjust:
        prices = extract_prices(data, field_preferred="Close", field_fallback="Close", tickers=tickers)
    else:
        prices = extract_prices(data, field_preferred="Adj Close", field_fallback="Close", tickers=tickers)

    # Czyszczenie: usuń kolumny niemal puste
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError("Po czyszczeniu brak danych cenowych.")
    min_non_na = max(50, int(0.4 * len(prices)))  # wymagaj min 50 obserwacji i >=40% próbki
    prices = prices.dropna(axis=1, thresh=min_non_na)
    prices = prices.sort_index(axis=1)
    print_step(f"Dane OK: kształt macierzy cen = {prices.shape}")
    return prices


def rolling_corr_last_window(prices: pd.DataFrame, a: str, b: str, lookback: int) -> Tuple[float, int]:
    """
    Liczy korelację Pearsona dziennych stóp zwrotu dla ostatniego okna lookback.
    Zwraca (corr, n_obs) — n_obs to liczba wspólnych obserwacji w oknie.
    """
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
    """
    Regressja OLS: y ~ alpha + beta * x  => zwraca (alpha, beta, spread=y - alpha - beta*x)
    """
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
    Szacuje half-life rewersji do średniej:
        Δs_t = ρ * s_{t-1} + ε_t
        half_life = -ln(2) / ln(1 + ρ)
    Zwraca NaN, gdy brak danych lub ρ bliskie -1.
    """
    s = spread.dropna()
    if len(s) < 40:
        return np.nan
    s_lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    # Dopasuj długości
    idx = s_lag.index.intersection(ds.index)
    y = ds.loc[idx].values
    x = s_lag.loc[idx].values
    if len(idx) < 30:
        return np.nan
    # OLS bez stałej
    rho = np.dot(x, y) / np.dot(x, x)
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
    """
    Liczy metryki pary (corr, pvalue, alpha, beta, half-life).
    coint_lookback: długość próbki dla kointegracji i hedge/half-life (ostatnie N dni wspólnych).
    """
    pa = prices[a].dropna()
    pb = prices[b].dropna()
    common = pa.index.intersection(pb.index)
    if len(common) < max(80, coint_lookback // 2):
        return None

    # CORR (na stopach zwrotu) — ostatnie 'corr_lookback' dni
    corr, corr_obs = rolling_corr_last_window(prices, a, b, corr_lookback)

    # Przygotuj zestaw do kointegracji/hedge — ostatnie 'coint_lookback' wspólnych dni
    pa2 = pa.loc[common].tail(coint_lookback)
    pb2 = pb.loc[common].tail(coint_lookback)

    # KOINTEGRACJA (Engle–Granger)
    # stat, pvalue, crit = coint(y, x)
    try:
        stat, pvalue, _ = coint(pa2, pb2)
    except Exception:
        return None

    # Hedge & spread & half-life
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
    coint_lookback: int = 250,
    corr_min: float = 0.85,
    pvalue_max: float = 0.03,
    half_life_max: float = 15,
    min_sample: int = 200,
    topk: int = 200
) -> pd.DataFrame:
    """
    Generuje metryki dla wszystkich par i aplikuje filtry jakości.
    Zwraca ramkę posortowaną: pvalue ASC, half-life ASC, corr DESC.
    """
    tickers = list(prices.columns)
    results: List[PairMetrics] = []

    pairs = list(itertools.combinations(tickers, 2))
    print_step(f"Liczenie metryk dla {len(pairs)} par (to może potrwać chwilę)...")

    for idx, (a, b) in enumerate(pairs, start=1):
        pm = compute_pair_metrics(prices, a, b, corr_lookback, coint_lookback)
        if pm is None:
            continue

        # Filtry twarde
        if not np.isfinite(pm.corr) or pm.corr < corr_min:
            continue
        if not np.isfinite(pm.pvalue) or pm.pvalue > pvalue_max:
            continue
        if not np.isfinite(pm.half_life) or pm.half_life > half_life_max:
            continue
        if pm.sample < min_sample:
            continue

        results.append(pm)

        # Prosty progress co 200 par
        if idx % 200 == 0:
            print_step(f"Przetworzono {idx}/{len(pairs)} par...")

    if not results:
        return pd.DataFrame(columns=[
            "a", "b", "corr", "corr_obs", "pvalue", "stat", "alpha", "beta", "half_life", "sample"
        ])

    df = pd.DataFrame([pm.__dict__ for pm in results])

    # Sortowanie jakościowe: pvalue rośnie (mniejsze lepsze), half-life rośnie (mniejsze lepsze), corr maleje (większe lepsze)
    df = df.sort_values(by=["pvalue", "half_life", "corr"], ascending=[True, True, False])
    if topk is not None and topk > 0:
        df = df.head(topk).reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Skanner par (XLK) z korelacją i kointegracją.")
    parser.add_argument("--tickers", nargs="*", help="Lista tickerów, np. --tickers MSFT AAPL NVDA")
    parser.add_argument("--tickers-file", default=None, help="Plik z tickerami (jeden na linię).")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help=f"Data początkowa (YYYY-MM-DD), domyślnie {DEFAULT_START_DATE}")
    parser.add_argument("--end-date", default=None, help="Data końcowa (YYYY-MM-DD), opcjonalnie.")
    parser.add_argument("--auto-adjust", action="store_true", help="Jeśli True, Close będzie już skorygowany (yfinance).")
    parser.add_argument("--corr-lookback", type=int, default=90, help="Okno do korelacji (na stopach zwrotu), domyślnie 90.")
    parser.add_argument("--coint-lookback", type=int, default=250, help="Okno do testu kointegracji i hedge/half-life, domyślnie 250.")
    parser.add_argument("--corr-min", type=float, default=0.85, help="Minimalna korelacja Pearsona (returns), domyślnie 0.85.")
    parser.add_argument("--pvalue-max", type=float, default=0.03, help="Maksymalne p-value (Engle–Granger), domyślnie 0.03.")
    parser.add_argument("--half-life-max", type=float, default=15, help="Maksymalny half-life spreadu (dni), domyślnie 15.")
    parser.add_argument("--min-sample", type=int, default=200, help="Minimalna wspólna próbka do kointegracji, domyślnie 200.")
    parser.add_argument("--topk", type=int, default=200, help="Ile najlepszych par zwrócić, domyślnie 200.")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help=f"Ścieżka wyjścia CSV, domyślnie {DEFAULT_OUT_CSV}")

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
        df = scan_pairs(
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

    # 4) Zapis CSV
    try:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
    except Exception as e:
        print(f"[ERROR] Zapis CSV nie powiódł się: {e}", file=sys.stderr)
        return 5

    print_step(f"Gotowe. Zapisano {len(df)} par -> {args.out_csv}")
    if len(df):
        print_step("Najlepsze 5 par:")
        print(df.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())