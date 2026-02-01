#!/usr/bin/env python3
import pandas as pd,glob,os

UNIVERSES=["tech_core","semis","software","financials","healthcare","discretionary"]

def main():
    frames=[]
    new_all=[]
    drop_all=[]
    for u in UNIVERSES:
        f=f"results/{u}/{u}_stable.csv"
        if os.path.exists(f):
            df=pd.read_csv(f)
            df["universe"]=u
            frames.append(df)
        newf=f"results/{u}_new_candidates.csv"
        if os.path.exists(newf): new_all.append(pd.read_csv(newf))
        dropf=f"results/{u}_dropped_pairs.csv"
        if os.path.exists(dropf): drop_all.append(pd.read_csv(dropf))

    comb=pd.concat(frames,ignore_index=True)
    comb.to_csv("results/combined_pairs_scored.csv",index=False)

    comb.sort_values(
        by=["grade","score_w"],
        ascending=[True,False]
    ).head(150).to_csv("results/combined_top150_watchlist.csv",index=False)

    if new_all:
        pd.concat(new_all).to_csv("results/new_candidates_global.csv",index=False)
    if drop_all:
        pd.concat(drop_all).to_csv("results/dropped_pairs_global.csv",index=False)

if __name__=="__main__":
    main()
