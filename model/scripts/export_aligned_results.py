"""Export aligned DesignB_plus_v1 noise and ablation results.

All input studies use the locked configuration from 实验配置与方法说明.md:
DI standard/100 epochs, AB micro/1 epoch, baselines per designB_reduced_plus,
boundary weight 0.0. Existing reduced-baseline tables are never overwritten.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT=Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/studies")
OUT=Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS=OUT/"表格分项归档"/"current_components"
CN={"di_emstgat":"DI-EMSTGAT（本文）","ab_emstgat":"AB-EMSTGAT（本文）","xgboost":"XGBoost","lightgbm":"LightGBM","patchtst":"PatchTST","mtgnn":"MTGNN","tcn":"TCN","transformer":"Transformer","itransformer":"iTransformer","gatv2":"GATv2"}
ORDER=["di_emstgat","ab_emstgat","patchtst","mtgnn","tcn","xgboost","lightgbm","itransformer","transformer","gatv2"]
AB_CONDS=[("clean","干净条件","ablation_designB_plus_v1_clean"),("snr10","SNR=10dB","ablation_designB_plus_v1_snr10"),("train20","训练集20%","ablation_designB_plus_v1_train20pct")]

def save(df,name):
 COMPONENTS.mkdir(parents=True,exist_ok=True); df.to_csv(COMPONENTS/(name+'.csv'),index=False,encoding='utf-8-sig'); print(name, len(df))
def load(d):
 p=ROOT/d/'per_seed_records.json'; return pd.DataFrame(json.loads(p.read_text(encoding='utf-8'))) if p.exists() else None

def main():
 noise=load('noise_designB_plus_v1')
 if noise is not None:
  grid=sorted({float(x) for x in noise.snr_db if x!='clean'},reverse=True); cols=['clean']+grid
  for metric,num in [('accuracy','表56_噪声鲁棒性_准确率随SNR_锁定配置'),('f1_macro','表57_噪声鲁棒性_宏F1随SNR_锁定配置'),('minority_recall','表58_噪声鲁棒性_最小类召回随SNR_锁定配置')]:
   rows=[]
   for m in ORDER:
    g=noise[noise.model==m]
    if g.empty: continue
    r={'模型':CN[m],'model_key':m,'种子数':int(g.seed.nunique())}
    for c in cols:
     z=g[g.snr_db==c]; tag='无噪' if c=='clean' else f'{int(c)}dB'; r[tag+'均值(%)']=round(100*z[metric].mean(),4); r[tag+'标准差(%)']=round(100*z[metric].std(ddof=1) if len(z)>1 else 0,4)
    rows.append(r)
   save(pd.DataFrame(rows),num)
  rows=[]
  for m in ORDER:
   g=noise[noise.model==m]
   if g.empty: continue
   clean=100*g[g.snr_db=='clean'].accuracy.mean(); vals=[100*g[g.snr_db==c].accuracy.mean() for c in grid]; drops=[clean-v for v in vals]
   r={'模型':CN[m],'model_key':m,'无噪准确率(%)':round(clean,4),'全网格平均准确率(%)':round(float(np.mean(vals)),4),'最大下降(个百分点)':round(float(max(drops)),4),'5dB准确率(%)':round(vals[-1],4)}
   for c,d in zip(grid,drops): r[f'{int(c)}dB_下降(个百分点)']=round(d,4)
   rows.append(r)
  # Ranking table is owned by scripts/finalize_aligned_reporting.py (表69/表70).
  print(pd.DataFrame(rows).sort_values('全网格平均准确率(%)',ascending=False).reset_index(drop=True).to_string(index=False))
  detail=noise.copy(); detail.insert(0,'模型',detail.model.map(CN));
  for c in ['accuracy','balanced_accuracy','f1_macro','f1_weighted','minority_recall']: detail[c]=(100*detail[c]).round(4)
  save(detail,'表60_噪声鲁棒性逐种子原始记录_锁定配置')

 all_ab=[]
 for key,label,dname in AB_CONDS:
  f=load(dname)
  if f is None: print('missing',dname); continue
  f=f.copy(); f['condition_key']=key; f['condition_cn']=label; all_ab.append(f)
 if not all_ab: return
 ab=pd.concat(all_ab,ignore_index=True)
 metrics=['accuracy','balanced_accuracy','f1_macro','f1_weighted','cohen_kappa']
 rows=[]
 for key,label,dname in AB_CONDS:
  f=ab[ab.condition_key==key]
  for model in ['di_emstgat','ab_emstgat']:
   s=f[f.model==model]; full=s[s.variant=='full'].set_index('seed').sort_index()
   for v in s.variant.unique():
    g=s[s.variant==v].set_index('seed').sort_index(); common=g.index.intersection(full.index)
    if g.empty or len(common)==0: continue
    r={'条件':label,'condition_key':key,'模型':CN[model],'model_key':model,'变体':g.variant_cn.iloc[0],'variant_key':v,'参数量':int(g.param_count.iloc[0]),'种子数':len(common)}
    for met in metrics: r[met+'均值(%)']=round(100*g[met].mean(),4); r[met+'标准差(%)']=round(100*g[met].std(ddof=1) if len(g)>1 else 0,4)
    if v=='full': r.update({'准确率变化(pp)':0.0,'宏F1变化(pp)':0.0,'配对t检验p':np.nan,'Wilcoxon p':np.nan,'变体更差种子':'—','结论':'基准（完整模型）'})
    else:
     da=(g.loc[common,'accuracy']-full.loc[common,'accuracy']).to_numpy(float); df1=(g.loc[common,'f1_macro']-full.loc[common,'f1_macro']).to_numpy(float)
     pt=1.0 if np.allclose(da,0) else float(ttest_rel(g.loc[common,'accuracy'],full.loc[common,'accuracy']).pvalue)
     try: pw=1.0 if np.allclose(da,0) else float(wilcoxon(da).pvalue)
     except ValueError: pw=np.nan
     r.update({'准确率变化(pp)':round(100*da.mean(),4),'宏F1变化(pp)':round(100*df1.mean(),4),'配对t检验p':round(pt,6),'Wilcoxon p':round(pw,6) if np.isfinite(pw) else np.nan,'变体更差种子':f'{int((da<0).sum())}/{len(da)}','结论':'待三条件综合判定'})
    rows.append(r)
 # Per-condition ablation statistics are owned by finalize_aligned_reporting.py (表71).
 summary=pd.DataFrame(rows); print(summary.head(20).to_string(index=False))
 detail=ab.copy(); detail.insert(0,'模型',detail.model.map(CN));
 for c in metrics: detail[c]=(100*detail[c]).round(4)
 save(detail,'表62_消融实验逐种子原始记录_锁定配置')
 # cross-condition delta/sign consistency
 rows=[]
 for model in ['di_emstgat','ab_emstgat']:
  for v in sorted(set(ab.variant)-{'full'}):
   r={'模型':CN[model],'model_key':model,'变体':VARIANT_CN.get(v,v),'variant_key':v}; signs=[]
   for key,label,dname in AB_CONDS:
    f=ab[(ab.model==model)&(ab.condition_key==key)]; full=f[f.variant=='full'].set_index('seed').accuracy.sort_index(); g=f[f.variant==v].set_index('seed').accuracy.sort_index(); common=g.index.intersection(full.index)
    if len(common):
     d=100*(g.loc[common]-full.loc[common]); r[label+'变化(pp)']=round(float(d.mean()),4); r[label+'p']=round(float(ttest_rel(g.loc[common],full.loc[common]).pvalue),6) if len(common)>1 and not np.allclose(d,0) else 1.0; signs.append(int(np.sign(d.mean())))
    else: r[label+'变化(pp)']=np.nan; r[label+'p']=np.nan
   nz=[s for s in signs if s]
   r['方向一致性']='一致：移除更好' if nz and all(s>0 for s in nz) else '一致：移除更差' if nz and all(s<0 for s in nz) else '条件间不一致' if nz else '未生效/无数据'; rows.append(r)
 # Direction consistency is owned by finalize_aligned_reporting.py (表72).
 cross=pd.DataFrame(rows); print(cross.to_string(index=False))
 # contract
 (OUT/'噪声与消融实验契约_锁定配置_v2.json').write_text(json.dumps({'noise_records':int(len(noise)) if noise is not None else 0,'ablation_conditions':{k:int(len(ab[ab.condition_key==k])) for k,_,_ in AB_CONDS if k in set(ab.condition_key)},'config':'DesignB_plus_v1 / designB_reduced_plus','historical_results_preserved':True},ensure_ascii=False,indent=2),encoding='utf-8')
 print('done')
VARIANT_CN={'no_dcc':'去除膨胀因果卷积','no_bigru':'去除双向GRU','no_knn_graph':'去除时序KNN图','no_self_attn':'去除多头自注意力','no_attn_pool':'注意力池化替为均值池化','no_skip_cls':'去除表格跳连分类器','no_pos_emb':'去除位置编码','no_boundary':'去除边界损失','no_router_gate':'去除特征相关性门控'}
if __name__=='__main__': main()
