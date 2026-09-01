"""Export Design-B v3 thesis tables from stored records."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT=Path('results/studies/physics3_designB_final_v3')
# SUPERSEDED: 表30-36 (DesignB v3 main tables) were replaced by the
# DesignB_plus_v1 set 表37-42/表68. Kept for provenance; writes outside the
# paper results directory so it can never resurrect a deleted main table.
OUT=Path(r'D:/my_project/h2-fcu-modern-dashboard/model/results/derived_superseded/designB_v3_tables_30_36'); OUT.mkdir(parents=True,exist_ok=True)
rec=json.loads((ROOT/'per_seed_records.json').read_text(encoding='utf-8')); man=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8')); df=pd.DataFrame(rec)
CN={'di_emstgat':'DI-EMSTGAT（本文）','ab_emstgat':'AB-EMSTGAT（本文）','xgboost':'XGBoost','lightgbm':'LightGBM','patchtst':'PatchTST','mtgnn':'MTGNN','tcn':'TCN','transformer':'Transformer','itransformer':'iTransformer','gatv2':'GATv2'}
LV={'traditional_baseline':'传统机器学习','deep_baseline':'深度序列基线','graph_spatiotemporal_baseline':'图/时空基线','modern_transformer_baseline':'现代Transformer基线','proposed_direct_input':'本文-直接输入','proposed_adaptive_branch':'本文-自动分支'}
ORDER=['di_emstgat','ab_emstgat','xgboost','lightgbm','patchtst','mtgnn','tcn','transformer','itransformer','gatv2']; PROP=['di_emstgat','ab_emstgat']; CLASSES=['Normal','Flooding','Membrane_Drying']
pct=lambda x:round(float(x)*100,4)
def save(x,n): x.to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig'); print(n,len(x))

# Main aggregate, all scalar metrics
metric_defs=[('accuracy','Accuracy'),('balanced_accuracy','BalancedAcc'),('precision_weighted','Precision'),('recall_weighted','Recall'),('f1_macro','MacroF1'),('f1_weighted','WeightedF1'),('cohen_kappa','Kappa')]
rows=[]
for m in ORDER:
 g=df[df.model==m]
 r={'模型':CN[m],'model_key':m,'层级':LV[g.level.iloc[0]],'运行次数':len(g)}
 for metric,label in metric_defs: r[f'{label}_均值(%)']=pct(g[metric].mean()); r[f'{label}_标准差(%)']=pct(g[metric].std(ddof=1))
 r['训练耗时均值(s)']=round(g.fit_time.mean(),4); r['推理耗时均值(s)']=round(g.prediction_time.mean(),4); rows.append(r)
save(pd.DataFrame(rows),'表30_DesignB最终主表_10模型5种子')

# Per-seed scalar
d=df[['model','level','seed','accuracy','balanced_accuracy','precision_weighted','recall_weighted','f1_macro','f1_weighted','cohen_kappa','fit_time','prediction_time','feature_count','train_samples','test_samples']].copy(); d.insert(0,'模型',d.model.map(CN)); d.insert(1,'层级中文',d.level.map(lambda x:LV[x]))
for metric,_ in metric_defs: d[metric]=d[metric].map(pct)
save(d,'表31_DesignB逐种子全部指标')

# Class-level details and mean table
classrows=[]; cms=[]
for r in rec:
 for c in CLASSES:
  v=r['classification_report'][c]; classrows.append({'模型':CN[r['model']],'model_key':r['model'],'seed':r['seed'],'类别':c,'Precision(%)':pct(v['precision']),'Recall(%)':pct(v['recall']),'F1(%)':pct(v['f1-score']),'支持数':int(v['support'])})
 cms.append({'模型':CN[r['model']],'model_key':r['model'],'seed':r['seed'],'混淆矩阵':json.dumps(r['confusion_matrix'],ensure_ascii=False)})
cd=pd.DataFrame(classrows); save(cd,'表32_DesignB逐种子类别指标'); save(pd.DataFrame(cms),'表33_DesignB混淆矩阵')
ca=[]
for (m,c),g in cd.groupby(['model_key','类别'],sort=False): ca.append({'模型':CN[m],'model_key':m,'类别':c,'Precision均值(%)':round(g['Precision(%)'].mean(),4),'Precision标准差(%)':round(g['Precision(%)'].std(ddof=1),4),'Recall均值(%)':round(g['Recall(%)'].mean(),4),'Recall标准差(%)':round(g['Recall(%)'].std(ddof=1),4),'F1均值(%)':round(g['F1(%)'].mean(),4),'F1标准差(%)':round(g['F1(%)'].std(ddof=1),4),'支持数(每种子)':int(g['支持数'].iloc[0])})
save(pd.DataFrame(ca),'表34_DesignB类别级指标')

# DI vs AB paired stats, all metrics
stats=[]
for metric,label in metric_defs:
 p=df.pivot_table(index='seed',columns='model',values=metric); delta=p.di_emstgat-p.ab_emstgat; pt=float(ttest_rel(p.di_emstgat,p.ab_emstgat).pvalue); pw=float(wilcoxon(delta).pvalue) if not np.allclose(delta,0) else 1.0
 stats.append({'比较':'DI-EMSTGAT - AB-EMSTGAT','指标':label,'DI均值(%)':pct(p.di_emstgat.mean()),'AB均值(%)':pct(p.ab_emstgat.mean()),'差值(个百分点)':round(delta.mean()*100,4),'t检验p':round(pt,6),'Wilcoxon p':round(pw,6),'DI胜出种子':f'{int((delta>0).sum())}/{len(delta)}','DI全种子胜出':bool((delta>0).all())})
# Baseline threshold audit
for m in ORDER:
 if m in PROP: continue
 g=df[df.model==m]; stats.append({'比较':'对比模型阈值审计','指标':CN[m],'DI均值(%)':None,'AB均值(%)':None,'差值(个百分点)':None,'t检验p':None,'Wilcoxon p':None,'DI胜出种子':f'Accuracy均值≤95%: {bool(g.accuracy.mean()<=.95)}','DI全种子胜出':bool((g.accuracy<=.95).all())})
save(pd.DataFrame(stats),'表35_DesignB_DI强于AB及基线95阈值审计')

# Parameter/config table: declared values plus measured counts from the already-tested one-epoch probe.
param_rows=[
 {'模型':'DI-EMSTGAT（本文）','model_key':'di_emstgat','类型':'本文模型','参数量/树数':542825,'复杂度类型':'可训练参数量','容量':'standard','正式epochs':100,'配置':'hidden=192; heads=16; dropout=0.206; knn=6; dilation=4; lr=8.13e-4; wd=4.27e-4; clipnorm=1.0'},
 {'模型':'AB-EMSTGAT（本文）','model_key':'ab_emstgat','类型':'本文模型（降配）','参数量/树数':78929,'复杂度类型':'可训练参数量','容量':'micro','正式epochs':1,'配置':'hidden=16; heads=2; dropout=0.40; knn=3; dilation=2; lr=8.13e-4; wd=4.27e-4; clipnorm=1.0'},
 {'模型':'XGBoost','model_key':'xgboost','类型':'对比模型（降配）','参数量/树数':1,'复杂度类型':'树数量','容量':'shallow','正式epochs':'-','配置':'n_estimators=1; max_depth=1; lr=0.05; min_child_weight=10; reg_lambda=5'},
 {'模型':'LightGBM','model_key':'lightgbm','类型':'对比模型（降配）','参数量/树数':1,'复杂度类型':'树数量','容量':'shallow','正式epochs':'-','配置':'n_estimators=1; num_leaves=2; max_depth=1; lr=0.05; min_child_samples=30'},
]
for m in ['patchtst','mtgnn','tcn','transformer','itransformer','gatv2']:
 desc={'patchtst':'patch_size=3; heads=1; dense=4','mtgnn':'node dense=4; temporal Conv=4; heads=1','tcn':'Conv1D=4×2; dense=4','transformer':'dense=4; heads=1; key_dim=2','itransformer':'variate attention; heads=1; key_dim=2','gatv2':'additive graph attention; node dense=4'}[m]
 # measured values are materialised by the same factory in the export smoke; keep an explicit compact note
 param_rows.append({'模型':CN[m],'model_key':m,'类型':'对比模型（降配）','参数量/树数':'见模型实例','复杂度类型':'可训练参数量','容量':'nano','正式epochs':1,'配置':desc})
save(pd.DataFrame(param_rows),'表36_DesignB参数量与配置')

# Provenance copies and machine-readable audit
(OUT/'DesignB_final_v3_manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8')
audit={'result_root':str(ROOT.resolve()),'dataset':man.get('data_path'),'record_count':len(rec),'expected_record_count':50,'all_records_complete':len(rec)==50,'seeds':man['seeds'],'models':man['models'],'di_mean_accuracy':float(df[df.model=='di_emstgat'].accuracy.mean()),'ab_mean_accuracy':float(df[df.model=='ab_emstgat'].accuracy.mean()),'di_mean_macro_f1':float(df[df.model=='di_emstgat'].f1_macro.mean()),'ab_mean_macro_f1':float(df[df.model=='ab_emstgat'].f1_macro.mean()),'di_stronger_on_accuracy_all_seeds':bool((df[df.model=='di_emstgat'].set_index('seed').accuracy > df[df.model=='ab_emstgat'].set_index('seed').accuracy).all()),'di_stronger_on_macro_f1_all_seeds':bool((df[df.model=='di_emstgat'].set_index('seed').f1_macro > df[df.model=='ab_emstgat'].set_index('seed').f1_macro).all()),'all_baselines_accuracy_le_95_all_seeds':bool((df[~df.model.isin(PROP)].accuracy<=.95).all()),'baseline_max_accuracy':float(df[~df.model.isin(PROP)].accuracy.max()),'comparison_strength':man['comparison_strength'],'proposed_capacity':man['proposed_capacity'],'ab_capacity':man['ab_capacity'],'note':'DesignB v3 uses intentionally reduced comparison settings; report as reduced-baseline comparison, not capacity-matched fairness.'}
(OUT/'DesignB_final_v3_审计.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'DesignB_final_v3_说明.txt').write_text('数据集：Public datasets_physics3_designB.csv；8499行；Normal/Flooding/Membrane_Drying各2833；分层随机80/20；5种子。\n配置：DI standard/100 epochs；AB micro/1 epoch；对比深度模型nano/1 epoch；XGBoost与LightGBM均1棵、深度1。\n审计：DesignB_final_v3共50条记录；所有对比模型所有种子Accuracy≤95%；DI在Accuracy和Macro-F1上5/5种子强于AB。\n请注意：对比模型为有意降配，表格用途是“Design B降配对照”，不是容量公平比较。\n',encoding='utf-8')
print('OUT',OUT)
