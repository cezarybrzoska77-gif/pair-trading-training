#!/usr/bin/env python3
"""
Retail Mode Pairs Scanner - XLK Universe
Skanuje pary pod kątem korelacji i kointegracji (Engle-Granger).
Autor: Senior Python/Quant DevOps
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint
from itertools import combinations


# Domyślny wszechświat XLK (35+ tickerów technologicznych)
DEFAULT_XLK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "ACN",
    "AMD", "INTC", "IBM", "INTU", "NOW", "TXN", "QCOM", "AMAT", "MU",
    "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MCHP", "FTNT", "PANW",
    "ANSS", "ADSK", "ROP", "PLTR", "CRWD", "SNOW", "ZS", "DDOG", "NET",
    "TEAM", "WDAY", "HUBS", "OKTA", "DXCM", "VEEV"
]


def load_tickers(tickers: Optional[List[str]], tickers_file: Optional[str]) -> List[str]:
    """Załaduj listę tickerów z argumentów CLI lub pliku."""
    if tickers_file:
        print(f"[INFO] Wczytuję tickery z pliku: {tickers_file}")
        with open(tickers_file, 'r') as f:
            file_tickers = [line.strip() for line in f if line.strip()]
        return file_tickers
    elif tickers:
        return tickers
    else:
        print(f"[INFO] Używam domyślnego wszechświata XLK ({len(DEFAULT_XLK_UNIVERSE)} tickerów)")
        return DEFAULT_XLK_UNIVERSE


def fetch_price_data(tickers: List[str], start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    """
    Pobierz dane cenowe z yfinance (Adj Close z fallbackiem na Close).
    Zwraca DataFrame z tickerami jako kolumnami.
    """
    print(f"[INFO] Pobieranie danych dla {len(tickers)} tickerów od {start_date}...")
    
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker',
            auto_adjust=False
        )
        
        if data.empty:
            print("[ERROR] Brak danych z yfinance. Sprawdź połączenie sieciowe i tickery.")
            sys.exit(1)
        
        # Wybór Adj Close z fallbackiem na Close
        prices = pd.DataFrame()
        for ticker in tickers:
            if len(tickers) == 1:
                # Pojedynczy ticker - inna struktura danych
                if 'Adj Close' in data.columns:
                    prices[ticker] = data['Adj Close']
                elif 'Close' in data.columns:
                    prices[ticker] = data['Close']
            else:
                # Wiele tickerów
                if ticker in data.columns:
                    if 'Adj Close' in data[ticker].columns:
                        prices[ticker] = data[ticker]['Adj Close']
                    elif 'Close' in data[ticker].columns:
                        prices[ticker] = data[ticker]['Close']
        
        # Usuń kolumny z >60% braków danych
        threshold = 0.4
        prices = prices.loc[:, prices.isnull().mean() < threshold]
        
        print(f"[INFO] Pobrano dane dla {len(prices.columns)} tickerów (po filtrowaniu braków)")
        print(f"[INFO] Zakres dat: {prices.index.min()} do {prices.index.max()}")
        
        if len(prices.columns) < 2:
            print("[ERROR] Za mało tickerów z wystarczającą ilością danych (minimum 2 wymagane)")
            sys.exit(1)
        
        return prices
    
    except Exception as e:
        print(f"[ERROR] Błąd pobierania danych: {e}")
        sys.exit(1)


def calculate_correlation(returns: pd.DataFrame, window: int) -> float:
    """Oblicz korelację Pearsona dla dwóch kolumn na ostatnich 'window' obserwacjach."""
    if len(returns) < window:
        return np.nan
    windowed = returns.tail(window)
    valid = windowed.dropna()
    if len(valid) < max(30, int(window * 0.5)):  # Min 30 obs lub 50% okna
        return np.nan
    return valid.iloc[:, 0].corr(valid.iloc[:, 1])


def calculate_cointegration(prices: pd.DataFrame, coint_window: int = 200) -> Tuple[float, float]:
    """
    Oblicz test kointegracji Engle-Granger na cenach.
    Zwraca (statystyka testowa, p-value).
    """
    if len(prices) < 100:
        return np.nan, np.nan
    
    windowed = prices.tail(coint_window)
    valid = windowed.dropna()
    
    if len(valid) < 100:
        return np.nan, np.nan
    
    try:
        stat, pvalue, _ = coint(valid.iloc[:, 0], valid.iloc[:, 1])
        return stat, pvalue
    except Exception:
        return np.nan, np.nan


def scan_pairs(
    prices: pd.DataFrame,
    corr_threshold: float,
    pvalue_max: float,
    min_sample: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Skanuj wszystkie pary tickerów pod kątem korelacji i kointegracji.
    Zwraca (kandydaci_df, wszystkie_metryki_df).
    """
    print("[INFO] Liczenie metryk dla wszystkich par...")
    
    tickers = prices.columns.tolist()
    all_pairs = list(combinations(tickers, 2))
    print(f"[INFO] Liczba par do przeskanowania: {len(all_pairs)}")
    
    results = []
    
    for ticker_a, ticker_b in all_pairs:
        pair_prices = prices[[ticker_a, ticker_b]].dropna()
        
        if len(pair_prices) < min_sample:
            continue
        
        # Oblicz stopy zwrotu
        returns = pair_prices.pct_change().dropna()
        
        # Korelacja w dwóch oknach
        corr_60 = calculate_correlation(returns, 60)
        corr_90 = calculate_correlation(returns, 90)
        
        # Liczba obserwacji w każdym oknie
        corr_obs_60 = min(len(returns), 60)
        corr_obs_90 = min(len(returns), 90)
        
        # Kointegracja
        coint_stat, coint_pvalue = calculate_cointegration(pair_prices, coint_window=200)
        
        results.append({
            'a': ticker_a,
            'b': ticker_b,
            'corr_60': corr_60,
            'corr_90': corr_90,
            'corr_obs_60': corr_obs_60,
            'corr_obs_90': corr_obs_90,
            'coint_stat': coint_stat,
            'coint_pvalue': coint_pvalue,
            'sample': len(pair_prices)
        })
    
    all_metrics = pd.DataFrame(results)
    
    print(f"[INFO] Obliczono metryki dla {len(all_metrics)} par")
    
    # Filtrowanie kandydatów
    candidates = all_metrics[
        (all_metrics['corr_60'] >= corr_threshold) &
        (all_metrics['corr_90'] >= corr_threshold) &
        (all_metrics['coint_pvalue'] <= pvalue_max) &
        (all_metrics['sample'] >= min_sample)
    ].copy()
    
    # Sortowanie: p-value ↑, corr_90 ↓, corr_60 ↓
    candidates = candidates.sort_values(
        by=['coint_pvalue', 'corr_90', 'corr_60'],
        ascending=[True, False, False]
    )
    
    print(f"[INFO] Znaleziono {len(candidates)} kandydatów spełniających kryteria")
    
    return candidates, all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Retail Mode Pairs Scanner - XLK Universe"
    )
    
    # Tickery
    parser.add_argument('--tickers', nargs='+', help='Lista tickerów do skanowania')
    parser.add_argument('--tickers-file', help='Plik z tickerami (jeden na linię)')
    
    # Daty
    parser.add_argument('--start-date', default='2018-01-01', help='Data początkowa (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Data końcowa (opcjonalnie)')
    
    # Filtry
    parser.add_argument('--corr-threshold', type=float, default=0.80, 
                       help='Próg korelacji (domyślnie 0.80)')
    parser.add_argument('--pvalue-max', type=float, default=0.10,
                       help='Maksymalne p-value kointegracji (domyślnie 0.10)')
    parser.add_argument('--min-sample', type=int, default=150,
                       help='Minimalna liczba wspólnych dni (domyślnie 150)')
    parser.add_argument('--topk', type=int, default=50,
                       help='Liczba top kandydatów do zapisania (domyślnie 50)')
    
    # Output
    parser.add_argument('--out-csv', default='results/xlk_pairs_candidates.csv',
                       help='Ścieżka do pliku CSV z kandydatami')
    
    # Kompatybilność wsteczna (mapowanie starych flag)
    parser.add_argument('--corr-min', type=float, dest='corr_threshold',
                       help='[PRZESTARZAŁE] Użyj --corr-threshold')
    parser.add_argument('--corr-lookback', type=int,
                       help='[PRZESTARZAŁE/IGNOROWANE] Nieużywane w tej wersji')
    
    args = parser.parse_args()
    
    # Wczytaj tickery
    tickers = load_tickers(args.tickers, args.tickers_file)
    
    # Pobierz dane
    prices = fetch_price_data(tickers, args.start_date, args.end_date)
    
    # Skanuj pary
    candidates, all_metrics = scan_pairs(
        prices,
        args.corr_threshold,
        args.pvalue_max,
        args.min_sample
    )
    
    # Utwórz katalog results
    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Zapisz kandydatów (top K)
    top_candidates = candidates.head(args.topk)
    top_candidates.to_csv(args.out_csv, index=False, float_format='%.6f')
    print(f"[INFO] Zapisano {len(top_candidates)} kandydatów do: {args.out_csv}")
    
    # Zapisz wszystkie metryki
    all_metrics_path = output_path.parent / 'xlk_pairs_all_metrics.csv'
    all_metrics.to_csv(all_metrics_path, index=False, float_format='%.6f')
    print(f"[INFO] Zapisano wszystkie metryki ({len(all_metrics)} par) do: {all_metrics_path}")
    
    if len(candidates) == 0:
        print("[WARNING] Brak par spełniających kryteria. Rozważ złagodzenie progów.")
        sys.exit(0)
    
    print("[INFO] Skanowanie zakończone pomyślnie!")


if __name__ == '__main__':
    main()