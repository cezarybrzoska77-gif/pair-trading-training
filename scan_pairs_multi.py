#!/usr/bin/env python3
import argparse
import os
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

warnings.filterwarnings("ignore")

UNIVERSE_CONFIG = {
    "tech_core": {"index": "QQQ"},
    "semis": {"index": "SOXX"},
    "software": {"index": "IGV"},
    "financials": {"index": "XLF"},
    "healthcare": {"index": "XLV"},
    "discretionary": {"index": "XLY"},
}

def parse_args():
    p = argparse.ArgumentParser("Multi-basket pair scanner")
    p.add_argument("--universe", required=True, choices=UNIVERSE_CONFIG.keys())
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--auto-adjust", action="store_true", default=True)
    p.add_argument("--use-percent-returns", action="store_true", default=False)
    p.add_argument("--winsorize", action="store_true", default=True)
    p.add_argument("--coint-lookbacks", default="240,300")
    p.add_argument("--min-sample", type=int, default=200)
    p.add_argument("--topk", type=int, default=100)
    return p.parse_args()

def load_tickers(universe):
    path = f"data/tickers_{universe}.txt"
    with open(path) as f:
        return [x.strip().upper() for x in f if x.strip()]

def download_prices(tickers, start, auto_adjust):
    data = yf.download(
        tickers,
        start=start,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    px = {}
    for t in tickers:
        try:
            s = data[(t, "Close")].dropna()
            if len(s) > 0:
                px[t] = s.rename(t)
        except Exception:
            continue
    return pd.concat(px.values(), axis=1)

def compute_returns(prices, percent=False):
    return prices.pct_change().dropna() if percent else np.log(prices).diff().dropna()

def winsorize(df):
    return df.clip(df.quantile(0.01), df.quantile(0.99), axis=1)

def residuals(price, index):
    df = pd.concat([price, index], axis=1).dropna()
    y = df.iloc[:,0]
    x = sm.add_constant(df.iloc[:,1])
    return sm.OLS(y, x).fit().resid

def rolling_spearman(a, b, w):
    df = pd.concat([a,b],axis=1).dropna()
    if len(df)<w: return np.nan
    return df.rank().rolling(w).corr().iloc[-1,0]

def engle_best(a,b,lbs):
    best=(np.nan,np.nan,np.nan)
    for lb in lbs:
        if len(a)>=lb and len(b)>=lb:
            try:
                stat,p,_=coint(a[-lb:],b[-lb:])
                if np.isnan(best[0]) or p<best[0]:
                    best=(p,stat,lb)
            except: pass
    return best

def main():
    args=parse_args()
    cfg=UNIVERSE_CONFIG[args.universe]
    tickers=load_tickers(args.universe)
    idx=cfg["index"]
    tickers=list(set(tickers+[idx]))

    outdir=f"results/{args.universe}"
    os.makedirs(outdir,exist_ok=True)

    prices=download_prices(tickers,args.start_date,args.auto_adjust)
    rets=compute_returns(prices,args.use_percent_returns)
    if args.winsorize: rets=winsorize(rets)

    lbs=[int(x) for x in args.coint_lookbacks.split(",")]
    rows=[]

    for a,b in itertools.combinations([t for t in rets.columns if t!=idx],2):
        df=pd.concat([rets[a],rets[b]],axis=1).dropna()
        if len(df)<args.min_sample: continue

        corr60=df[a].rolling(60).corr(df[b]).iloc[-1]
        corr90=df[a].rolling(90).corr(df[b]).iloc[-1]
        corr_mean=np.nanmean([corr60,corr90])

        sp60=rolling_spearman(df[a],df[b],60)
        sp90=rolling_spearman(df[a],df[b],90)

        ra=residuals(prices[a],prices[idx])
        rb=residuals(prices[b],prices[idx])
        rdf=pd.concat([ra,rb],axis=1).dropna()
        r60=rdf.iloc[:,0].rolling(60).corr(rdf.iloc[:,1]).iloc[-1]
        r90=rdf.iloc[:,0].rolling(90).corr(rdf.iloc[:,1]).iloc[-1]
        rmax=np.nanmax([r60,r90])

        hit=(df[a].rolling(30).corr(df[b])>=0.8).iloc[-126:].mean()

        pval,stat,lb=engle_best(prices[a],prices[b],lbs)

        grade=None
        if corr_mean>=0.82 and pval<=0.05 and hit>=0.70 and rmax>=0.60:
            grade="A"
        elif ((corr_mean>=0.78 or (corr60>=0.80 and corr90>=0.76))
              and pval<=0.12 and sp60>=0.75 and sp90>=0.75 and hit>=0.70 and rmax>=0.60):
            grade="B+"

        rows.append({
            "universe":args.universe,"a":a,"b":b,"pair":f"{a}/{b}",
            "corr_60":corr60,"corr_90":corr90,"corr_mean":corr_mean,
            "spearman_60":sp60,"spearman_90":sp90,
            "resid_corr_max":rmax,"corr_hitrate_30d_6m":hit,
            "coint_pvalue_best":pval,"coint_stat_best":stat,"coint_lookback_best":lb,
            "grade":grade
        })

    df=pd.DataFrame(rows)
    df.to_csv(f"{outdir}/{args.universe}_all_metrics.csv",index=False)
    df[df.grade.notna()].to_csv(f"{outdir}/{args.universe}_candidates.csv",index=False)

if __name__=="__main__":
    main()
