#!/usr/bin/env python3
import argparse,os,pandas as pd
from datetime import datetime

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--universe",required=True)
    return p.parse_args()

def main():
    a=parse()
    today=datetime.utcnow().date().isoformat()
    incsv=f"results/{a.universe}/{a.universe}_candidates.csv"
    statef=f"state/{a.universe}_history.csv"
    outdir=f"results/{a.universe}"
    os.makedirs("state",exist_ok=True)

    cur=pd.read_csv(incsv)
    cur["pair"]=cur["a"]+"/"+cur["b"]

    if os.path.exists(statef):
        st=pd.read_csv(statef)
    else:
        st=pd.DataFrame(columns=["pair","persistence_days","drop_streak","first_seen","last_seen"])

    st=st.set_index("pair",drop=False)

    stable,new,dropped=[],[],[]

    for _,r in cur.iterrows():
        p=r["pair"]
        if p not in st.index:
            st.loc[p]=[p,0,0,today,today]

        s=st.loc[p]
        entry=r["grade"] in ["A","B+"]
        stay=r["corr_mean"]>=0.78 and r["coint_pvalue_best"]<=0.14
        drop=r["corr_mean"]<0.75 or r["coint_pvalue_best"]>0.18

        if entry or stay:
            s.persistence_days+=1
            s.drop_streak=0
            s.last_seen=today
        elif drop:
            s.drop_streak+=1

        st.loc[p]=s

        if entry and s.persistence_days==5:
            new.append({**r,"persistence_days":s.persistence_days})

        if s.drop_streak>=3:
            dropped.append({"pair":p,"last_seen":today})
            st.drop(p,inplace=True)
            continue

        if entry and s.persistence_days>=5:
            stable.append({**r,"persistence_days":s.persistence_days})

    st.reset_index(drop=True).to_csv(statef,index=False)
    pd.DataFrame(stable).to_csv(f"{outdir}/{a.universe}_stable.csv",index=False)
    pd.DataFrame(new).to_csv(f"results/{a.universe}_new_candidates.csv",index=False)
    pd.DataFrame(dropped).to_csv(f"results/{a.universe}_dropped_pairs.csv",index=False)

if __name__=="__main__":
    main()
