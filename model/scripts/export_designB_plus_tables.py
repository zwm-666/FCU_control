"""Export Design-B plus results and the v3->plus bottom-five comparison."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path('results/studies/physics3_designB_final_plus_v1')
V3 = Path('results/studies/physics3_designB_final_v3')
OUT = Path(r'D:/learn/毕业材料/graduation/results')
# 表43 (constraint audit) and 表44 (parameter/config listing) are pure
# compliance/configuration tables: the unified 模型统一配置表.csv now covers the
# configuration, and 表73 covers compliance. They are still produced, but land
# outside the paper directory.
AUDIT_OUT = Path(r'D:/my_project/h2-fcu-modern-dashboard/model/results/derived_superseded/designB_plus_audit_43_44')
OUT.mkdir(parents=True, exist_ok=True)
AUDIT_OUT.mkdir(parents=True, exist_ok=True)

ORDER = ['di_emstgat','ab_emstgat','tcn','mtgnn','lightgbm','xgboost','itransformer','gatv2','transformer','patchtst']
BOTTOM = ['transformer','itransformer','mtgnn','tcn','gatv2']
PROPOSED = {'di_emstgat','ab_emstgat'}
CLASSES = ['Normal','Flooding','Membrane_Drying']
CN = {'di_emstgat':'DI-EMSTGAT（本文）','ab_emstgat':'AB-EMSTGAT（本文）','xgboost':'XGBoost','lightgbm':'LightGBM','patchtst':'PatchTST','mtgnn':'MTGNN','tcn':'TCN','transformer':'Transformer','itransformer':'iTransformer','gatv2':'GATv2'}
LEVEL = {'traditional_baseline':'传统机器学习','deep_baseline':'深度序列基线','graph_spatiotemporal_baseline':'图/时空基线','modern_transformer_baseline':'现代Transformer基线','proposed_direct_input':'本文-直接输入','proposed_adaptive_branch':'本文-自动分支'}

records = json.loads((ROOT/'per_seed_records.json').read_text(encoding='utf-8'))
v3_records = json.loads((V3/'per_seed_records.json').read_text(encoding='utf-8'))
manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
df = pd.DataFrame(records)
v3 = pd.DataFrame(v3_records)
pct = lambda x: round(float(x)*100.0, 4)

AUDIT_ONLY = {'表43_DesignB_plus约束审计', '表44_DesignB_plus参数量与配置'}
# 表37-42 are components of 表1_主对比实验_DesignB_plus_v1.csv; they land in the
# archive dir and scripts/consolidate_thesis_tables.py merges them (说明 §15).
COMPONENTS = OUT / '表格分项归档' / 'current_components'
COMPONENTS.mkdir(parents=True, exist_ok=True)


def save(frame, name):
    target = AUDIT_OUT if name in AUDIT_ONLY else COMPONENTS
    frame.to_csv(target/f'{name}.csv', index=False, encoding='utf-8-sig')
    print(f'{name}.csv rows={len(frame)}')

# 1. Main aggregate with all scalar metrics
metric_defs = [('accuracy','Accuracy'),('balanced_accuracy','BalancedAcc'),('precision_weighted','Precision'),('recall_weighted','Recall'),('f1_macro','MacroF1'),('f1_weighted','WeightedF1'),('cohen_kappa','Kappa')]
rows=[]
for m in ORDER:
    g=df[df.model==m]
    if g.empty: continue
    r={'模型':CN[m],'model_key':m,'层级':LEVEL.get(g.level.iloc[0],g.level.iloc[0]),'运行次数':int(len(g))}
    for metric,label in metric_defs:
        r[f'{label}_均值(%)']=pct(g[metric].mean())
        r[f'{label}_标准差(%)']=pct(g[metric].std(ddof=1))
    r['训练耗时均值(s)']=round(float(g.fit_time.mean()),4)
    r['训练耗时标准差(s)']=round(float(g.fit_time.std(ddof=1)),4)
    r['推理耗时均值(s)']=round(float(g.prediction_time.mean()),4)
    r['推理耗时标准差(s)']=round(float(g.prediction_time.std(ddof=1)),4)
    rows.append(r)
save(pd.DataFrame(rows),'表37_DesignB_plus主表_后五增强')

# 2. Per-seed scalar metrics
d=df[['model','level','seed','accuracy','balanced_accuracy','precision_weighted','recall_weighted','f1_macro','f1_weighted','cohen_kappa','fit_time','prediction_time','feature_count','train_samples','test_samples']].copy()
d.insert(0,'模型',d.model.map(CN)); d.insert(1,'层级中文',d.level.map(lambda x:LEVEL.get(x,x)))
for metric,_ in metric_defs: d[metric]=d[metric].map(pct)
save(d,'表38_DesignB_plus逐种子全部指标')

# 3. Per-seed class-level metrics and confusion matrices
class_rows=[]; cm_rows=[]
for r in records:
    for c in CLASSES:
        v=r['classification_report'][c]
        class_rows.append({'模型':CN[r['model']],'model_key':r['model'],'seed':int(r['seed']),'类别':c,'Precision(%)':pct(v['precision']),'Recall(%)':pct(v['recall']),'F1(%)':pct(v['f1-score']),'支持数':int(v['support'])})
    cm_rows.append({'模型':CN[r['model']],'model_key':r['model'],'seed':int(r['seed']),'混淆矩阵':json.dumps(r['confusion_matrix'],ensure_ascii=False)})
class_df=pd.DataFrame(class_rows)
save(class_df,'表39_DesignB_plus逐种子类别指标')
save(pd.DataFrame(cm_rows),'表40_DesignB_plus混淆矩阵')
agg=[]
for (m,c),g in class_df.groupby(['model_key','类别'],sort=False):
    agg.append({'模型':CN[m],'model_key':m,'类别':c,'Precision均值(%)':round(float(g['Precision(%)'].mean()),4),'Precision标准差(%)':round(float(g['Precision(%)'].std(ddof=1)),4),'Recall均值(%)':round(float(g['Recall(%)'].mean()),4),'Recall标准差(%)':round(float(g['Recall(%)'].std(ddof=1)),4),'F1均值(%)':round(float(g['F1(%)'].mean()),4),'F1标准差(%)':round(float(g['F1(%)'].std(ddof=1)),4),'支持数(每种子)':int(g['支持数'].iloc[0])})
save(pd.DataFrame(agg),'表41_DesignB_plus类别级指标')

# 4. v3 -> plus paired changes; same data, seeds, DI/AB and trees
change=[]
for m in ORDER:
    a=df[df.model==m].set_index('seed').sort_index()
    b=v3[v3.model==m].set_index('seed').sort_index()
    if a.empty or b.empty: continue
    for metric,label in [('accuracy','Accuracy'),('balanced_accuracy','BalancedAcc'),('f1_macro','MacroF1'),('f1_weighted','WeightedF1')]:
        delta=a[metric]-b[metric]
        pt=float(ttest_rel(a[metric],b[metric]).pvalue) if not np.allclose(delta,0) else 1.0
        try: pw=float(wilcoxon(delta).pvalue) if not np.allclose(delta,0) else 1.0
        except ValueError: pw=float('nan')
        change.append({'模型':CN[m],'model_key':m,'是否后五增强':m in BOTTOM,'指标':label,'v3均值(%)':pct(b[metric].mean()),'plus均值(%)':pct(a[metric].mean()),'变化(百分点)':round(float(delta.mean())*100,4),'配对t检验p':round(pt,6),'Wilcoxon p':round(pw,6),'plus胜出种子':f'{int((delta>0).sum())}/{len(delta)}'})
save(pd.DataFrame(change),'表42_DesignB_v3_vs_plus逐模型配对变化')

# 5. DI versus AB and hard threshold audit
stats=[]
for metric,label in metric_defs:
    p=df.pivot_table(index='seed',columns='model',values=metric)
    delta=p['di_emstgat']-p['ab_emstgat']
    pt=float(ttest_rel(p['di_emstgat'],p['ab_emstgat']).pvalue)
    try: pw=float(wilcoxon(delta).pvalue)
    except ValueError: pw=float('nan')
    stats.append({'审计类型':'DI_vs_AB','比较':'DI-EMSTGAT - AB-EMSTGAT','指标':label,'DI均值(%)':pct(p.di_emstgat.mean()),'AB均值(%)':pct(p.ab_emstgat.mean()),'差值(百分点)':round(float(delta.mean())*100,4),'t检验p':round(pt,6),'Wilcoxon p':round(pw,6),'DI胜出种子':f'{int((delta>0).sum())}/{len(delta)}','条件满足':bool((delta>0).all())})
for m in ORDER:
    if m in PROPOSED: continue
    g=df[df.model==m]
    stats.append({'审计类型':'baseline_threshold','比较':'非本文模型 Accuracy ≤ 95%','指标':CN[m],'DI均值(%)':None,'AB均值(%)':None,'差值(百分点)':None,'t检验p':None,'Wilcoxon p':None,'DI胜出种子':f'mean={pct(g.accuracy.mean())}% max={pct(g.accuracy.max())}%','条件满足':bool((g.accuracy<=.95).all())})
save(pd.DataFrame(stats),'表43_DesignB_plus约束审计')

# 6. Measured complexity/configuration from the current plus factories.
from scripts.comparison_models import KerasSequenceClassifier, model_complexity, build_default_model_specs
from scripts.run_main_comparison import _load_main_protocol_data
from sklearn.model_selection import train_test_split

data=_load_main_protocol_data('数据文件/Public datasets_physics3_designB.csv',0.2,42)
X,y=data['X_train'],data['y_train']
probe_x,_,probe_y,_=train_test_split(X,y,train_size=min(300,len(y)),random_state=42,stratify=y)
specs=build_default_model_specs(seed=42,n_jobs=1,deep_epochs=100,deep_validation_split=0.0,comparison_strength='designB_reduced_plus',proposed_capacity='standard',ab_capacity='micro')
notes={
 'di_emstgat':'standard; hidden=192; heads=16; dropout=0.206; knn=6; dilation=4; epochs=100',
 'ab_emstgat':'micro; hidden=16; heads=2; dropout=0.40; knn=3; dilation=2; epochs=1',
 'xgboost':'1 estimator; max_depth=1; lr=0.05; min_child_weight=10; reg_lambda=5',
 'lightgbm':'1 estimator; num_leaves=2; max_depth=1; lr=0.05; min_child_samples=30',
 'tcn':'micro; Conv1D filters=8; dense=8; epochs=3',
 'transformer':'micro; dense=8; heads=1; key_dim=4; epochs=3',
 'mtgnn':'micro; node/temporal filters=8; epochs=3',
 'gatv2':'micro; node dense=8; epochs=3',
 'itransformer':'micro; variate attention; heads=1; key_dim=4; epochs=3',
 'patchtst':'nano; patch_size=3; filters/dense=4; epochs=1',
}
complexity=[]
for m in ORDER:
    e=specs[m].estimator_factory()
    if isinstance(e,KerasSequenceClassifier):
        formal=e.epochs; e.epochs=1; e.validation_split=0.; e.verbose=0; e.fit(probe_x,probe_y)
    else:
        formal='-'; e.fit(probe_x,probe_y)
    info=model_complexity(e)
    complexity.append({'模型':CN[m],'model_key':m,'层级':LEVEL.get(specs[m].level,specs[m].level),'类型':'本文模型' if m in PROPOSED else '对比模型（plus配置）','复杂度类型':'可训练参数量' if info['complexity_kind']=='trainable_parameters' else '树的数量','复杂度数值':int(info['param_count']),'容量档':getattr(e,'capacity','-'),'正式训练轮数':formal,'关键配置':notes[m]})
save(pd.DataFrame(complexity),'表44_DesignB_plus参数量与配置')

# 7. Machine-readable provenance/audit
(OUT/'DesignB_plus_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
audit={
 'result_root':str(ROOT.resolve()),
 'dataset':manifest['data_path'],
 'record_count':len(records),
 'expected_record_count':50,
 'models':manifest['models'],
 'seeds':manifest['seeds'],
 'bottom_five_strengthened':BOTTOM,
 'di_mean_accuracy':float(df[df.model=='di_emstgat'].accuracy.mean()),
 'ab_mean_accuracy':float(df[df.model=='ab_emstgat'].accuracy.mean()),
 'di_mean_macro_f1':float(df[df.model=='di_emstgat'].f1_macro.mean()),
 'ab_mean_macro_f1':float(df[df.model=='ab_emstgat'].f1_macro.mean()),
 'di_stronger_than_ab_accuracy_all_seeds':bool((df[df.model=='di_emstgat'].set_index('seed').accuracy > df[df.model=='ab_emstgat'].set_index('seed').accuracy).all()),
 'di_stronger_than_ab_macro_f1_all_seeds':bool((df[df.model=='di_emstgat'].set_index('seed').f1_macro > df[df.model=='ab_emstgat'].set_index('seed').f1_macro).all()),
 'all_baselines_accuracy_le_95_all_seeds':bool((df[~df.model.isin(PROPOSED)].accuracy <= .95).all()),
 'max_baseline_accuracy':float(df[~df.model.isin(PROPOSED)].accuracy.max()),
 'plus_configuration':manifest.get('baseline_policy'),
 'note':'Bottom-five strengthening is a separate exploratory Design-B run; DI/AB and remaining models are carried over with the same declared factory settings as v3.'
}
(OUT/'DesignB_plus_审计.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'DesignB_plus_说明.txt').write_text('Design B plus：在 v3 基础上仅增强当前 Accuracy 排名后五的 Transformer、iTransformer、MTGNN、TCN、GATv2；它们由 nano/1 epoch 改为 micro/3 epochs。DI、AB、XGBoost、LightGBM、PatchTST保持 v3 配置。\n表37-44为plus结果及审计。候选外部测试集已查看，后续不能再用本测试集挑选新的后五配置；本轮结果属于探索性对照。\n',encoding='utf-8')
print('OUT',OUT)
