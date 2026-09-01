"""SUPERSEDED noise exporter (表48-52).

Its tables were byte-identical duplicates of 表56-60, so the paper directory now
keeps only the 表56-60 / 表69 / 表70 set produced by
``scripts/export_noise_ablation_tables.py`` and
``scripts/finalize_aligned_reporting.py``. This script is kept for provenance
and now writes to ``model/results/derived_superseded/``, never to the paper
results directory.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/studies/noise_designB_plus_v1")
OUT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/derived_superseded/locked_noise_48_52")
ORDER = ["di_emstgat", "ab_emstgat", "patchtst", "mtgnn", "tcn", "xgboost", "lightgbm", "itransformer", "transformer", "gatv2"]
CN = {"di_emstgat":"DI-EMSTGAT（本文）","ab_emstgat":"AB-EMSTGAT（本文）","xgboost":"XGBoost","lightgbm":"LightGBM","patchtst":"PatchTST","mtgnn":"MTGNN","tcn":"TCN","transformer":"Transformer","itransformer":"iTransformer","gatv2":"GATv2"}

def load():
    return pd.DataFrame(json.loads((ROOT / "per_seed_records.json").read_text(encoding="utf-8")))

def cols(frame):
    grid = sorted({float(v) for v in frame.snr_db if v != "clean"}, reverse=True)
    return ["clean"] + grid

def save(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(name, len(frame))

def metric_table(frame, metric):
    result=[]
    for model in ORDER:
        g=frame[frame.model==model]
        if g.empty: continue
        row={"模型":CN[model],"model_key":model,"种子数":int((g.snr_db=="clean").sum())}
        for snr in cols(frame):
            s=g[g.snr_db==snr]
            tag="无噪" if snr=="clean" else f"{int(snr)}dB"
            row[f"{tag}_均值(%)"]=round(float(s[metric].mean()*100),4)
            row[f"{tag}_标准差(%)"]=round(float(s[metric].std(ddof=1)*100) if len(s)>1 else 0.0,4)
        result.append(row)
    return pd.DataFrame(result)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame=load()
    expected=10*5*9
    if len(frame)!=expected: raise AssertionError(f"expected {expected} records, got {len(frame)}")
    save(metric_table(frame,"accuracy"),"表48_噪声鲁棒性_准确率随SNR_锁定配置")
    save(metric_table(frame,"f1_macro"),"表49_噪声鲁棒性_宏F1随SNR_锁定配置")
    save(metric_table(frame,"minority_recall"),"表50_噪声鲁棒性_最小类召回随SNR_锁定配置")
    grid=[x for x in cols(frame) if x!="clean"]
    rows=[]
    for model in ORDER:
        g=frame[frame.model==model]
        if g.empty: continue
        clean=float(g[g.snr_db=="clean"].accuracy.mean()*100)
        drops=[]
        row={"模型":CN[model],"model_key":model,"无噪准确率(%)":round(clean,4)}
        for snr in grid:
            acc=float(g[g.snr_db==snr].accuracy.mean()*100)
            drop=clean-acc; drops.append(drop); row[f"{int(snr)}dB_下降(个百分点)"]=round(drop,4)
        row["全网格平均准确率(%)"]=round(float(np.mean([float(g[g.snr_db==snr].accuracy.mean()*100) for snr in grid])),4)
        row["最大下降(个百分点)"]=round(float(np.max(drops)),4)
        row["5dB准确率(%)"]=round(float(g[g.snr_db==min(grid)].accuracy.mean()*100),4)
        row["鲁棒性排序依据"]="全网格平均准确率越高越鲁棒"
        rows.append(row)
    deg=pd.DataFrame(rows).sort_values("全网格平均准确率(%)",ascending=False).reset_index(drop=True)
    deg.insert(0,"鲁棒性排名",np.arange(1,len(deg)+1))
    save(deg,"表51_噪声退化与鲁棒性排名_锁定配置")
    detail=frame.copy(); detail.insert(0,"模型",detail.model.map(CN))
    for c in ["accuracy","balanced_accuracy","f1_macro","f1_weighted","minority_recall"]:
        detail[c]=(detail[c]*100).round(4)
    save(detail,"表52_噪声鲁棒性逐种子原始记录_锁定配置")
    print(json.dumps({"source":str(ROOT),"records":len(frame),"models":sorted(frame.model.unique().tolist()),"seeds":sorted(frame.seed.unique().tolist()),"snr_levels":cols(frame)},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
