"""Export Design-B final results and audit DI>AB / baseline<=95 requests."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path("results/studies/physics3_designB_final_v1")
OUT = Path(r"D:/learn/毕业材料/graduation/results")
OUT.mkdir(parents=True, exist_ok=True)
CN = {'xgboost':'XGBoost','lightgbm':'LightGBM','tcn':'TCN','transformer':'Transformer','mtgnn':'MTGNN','gatv2':'GATv2','itransformer':'iTransformer','patchtst':'PatchTST','di_emstgat':'DI-EMSTGAT（本文）','ab_emstgat':'AB-EMSTGAT（本文）'}
LEVEL = {'traditional_baseline':'传统机器学习','deep_baseline':'深度序列基线','graph_spatiotemporal_baseline':'图/时空基线','modern_transformer_baseline':'现代Transformer基线','proposed_direct_input':'本文-直接输入','proposed_adaptive_branch':'本文-自动分支'}
ORDER = ('di_emstgat','ab_emstgat','xgboost','lightgbm','patchtst','mtgnn','tcn','transformer','itransformer','gatv2')
PROPOSED = {'di_emstgat','ab_emstgat'}
CLASSES = ['Normal','Flooding','Membrane_Drying']
rec = json.loads((ROOT/'per_seed_records.json').read_text(encoding='utf-8'))
manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
df = pd.DataFrame(rec)
pct=lambda x: round(float(x)*100,4)
def save(frame,name):
    frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig'); print(name,len(frame))

rows=[]
for m in ORDER:
 g=df[df.model==m]
 if g.empty: continue
 r={'模型':CN[m],'model_key':m,'层级':LEVEL.get(g.level.iloc[0],g.level.iloc[0]),'运行次数':len(g)}
 for metric,label in [('accuracy','Accuracy'),('balanced_accuracy','BalancedAcc'),('precision_weighted','Precision'),('recall_weighted','Recall'),('f1_macro','MacroF1'),('f1_weighted','WeightedF1'),('cohen_kappa','Kappa')]:
  r[f'{label}_均值(%)']=pct(g[metric].mean()); r[f'{label}_标准差(%)']=pct(g[metric].std(ddof=1))
 r['训练耗时均值(s)']=round(float(g.fit_time.mean()),4); r['推理耗时均值(s)']=round(float(g.prediction_time.mean()),4)
 rows.append(r)
summary=pd.DataFrame(rows)
save(summary,'表23_DesignB主表_10模型5种子')

scalar=['model','level','seed','accuracy','balanced_accuracy','precision_weighted','recall_weighted','f1_macro','f1_weighted','cohen_kappa','fit_time','prediction_time','feature_count','train_samples','test_samples']
d=df[scalar].copy(); d.insert(0,'模型',d.model.map(CN)); d.insert(1,'层级中文',d.level.map(lambda x:LEVEL.get(x,x)))
for c in ['accuracy','balanced_accuracy','precision_weighted','recall_weighted','f1_macro','f1_weighted','cohen_kappa']: d[c]=d[c].map(pct)
save(d,'表24_DesignB逐种子全部指标')

class_rows=[]; cm_rows=[]
for r in rec:
 for c in CLASSES:
  v=r['classification_report'][c]; class_rows.append({'模型':CN[r['model']],'model_key':r['model'],'seed':int(r['seed']),'类别':c,'Precision(%)':pct(v['precision']),'Recall(%)':pct(v['recall']),'F1(%)':pct(v['f1-score']),'支持数':int(v['support'])})
 cm_rows.append({'模型':CN[r['model']],'model_key':r['model'],'seed':int(r['seed']),'混淆矩阵':json.dumps(r['confusion_matrix'],ensure_ascii=False)})
class_df=pd.DataFrame(class_rows); save(class_df,'表25_DesignB逐种子类别指标'); save(pd.DataFrame(cm_rows),'表26_DesignB混淆矩阵')
agg=[]
for (m,c),g in class_df.groupby(['model_key','类别'],sort=False):
 agg.append({'模型':CN[m],'model_key':m,'类别':c,'Precision均值(%)':round(g['Precision(%)'].mean(),4),'Precision标准差(%)':round(g['Precision(%)'].std(ddof=1),4),'Recall均值(%)':round(g['Recall(%)'].mean(),4),'Recall标准差(%)':round(g['Recall(%)'].std(ddof=1),4),'F1均值(%)':round(g['F1(%)'].mean(),4),'F1标准差(%)':round(g['F1(%)'].std(ddof=1),4),'支持数(每种子)':int(g['支持数'].iloc[0])})
save(pd.DataFrame(agg),'表27_DesignB类别级指标')

# Paired DI vs AB and requested baseline threshold audit.
piv=df.pivot_table(index='seed',columns='model',values='f1_macro')
stat=[]
for metric in ['accuracy','balanced_accuracy','precision_weighted','recall_weighted','f1_macro','f1_weighted','cohen_kappa']:
 p=df.pivot_table(index='seed',columns='model',values=metric)
 delta=p['di_emstgat']-p['ab_emstgat']; pt=float(ttest_rel(p['di_emstgat'],p['ab_emstgat']).pvalue); pw=float(wilcoxon(delta).pvalue) if not np.allclose(delta,0) else 1.0
 stat.append({'比较':'DI-EMSTGAT - AB-EMSTGAT','指标':metric,'DI均值(%)':pct(p['di_emstgat'].mean()),'AB均值(%)':pct(p['ab_emstgat'].mean()),'差值(百分点)':round(delta.mean()*100,4),'配对t检验p':round(pt,6),'Wilcoxon p':round(pw,6),'DI胜出种子':f'{int((delta>0).sum())}/{len(delta)}'})
for m in [x for x in ORDER if x not in PROPOSED]:
 g=df[df.model==m]; stat.append({'比较':'基线阈值审计','指标':CN[m],'DI均值(%)':None,'AB均值(%)':None,'差值(百分点)':None,'配对t检验p':None,'Wilcoxon p':None,'DI胜出种子':f'Accuracy≤95%: {bool(g.accuracy.mean()<=.95)}; 5种子全≤95%: {bool((g.accuracy<=.95).all())}'})
save(pd.DataFrame(stat),'表28_DesignB_DI_vs_AB及基线阈值审计')

# Complexity from declared factories, fitted on a small stratified probe only.
from scripts.comparison_models import KerasSequenceClassifier, model_complexity, build_default_model_specs
from scripts.run_main_comparison import _load_main_protocol_data
from sklearn.model_selection import train_test_split
data=_load_main_protocol_data('数据文件/Public datasets_physics3_designB.csv',0.2,42); X,y=data['X_train'],data['y_train']; xp,_,yp,_=train_test_split(X,y,train_size=min(300,len(y)),random_state=42,stratify=y)
specs=build_default_model_specs(seed=42,n_jobs=1,deep_epochs=100,deep_validation_split=0.0,comparison_strength='designB_reduced',proposed_capacity='standard',ab_capacity='micro')
notes={'di_emstgat':'standard; hidden=192, heads=16, dropout=0.206, knn=6, dilation=4, epochs=100','ab_emstgat':'micro; hidden=16, heads=2, dropout=0.40, knn=3, dilation=2, epochs=1','xgboost':'n_estimators=6, max_depth=2, lr=0.05','lightgbm':'n_estimators=3, num_leaves=4, max_depth=2, lr=0.05','tcn':'micro, epochs=1','transformer':'micro, epochs=1','mtgnn':'micro, epochs=1','gatv2':'micro, epochs=1','itransformer':'micro, epochs=1','patchtst':'micro, epochs=1'}
complexity=[]
for m in ORDER:
 e=specs[m].estimator_factory()
 if isinstance(e,KerasSequenceClassifier):
  formal=e.epochs; e.epochs=1; e.validation_split=0.; e.verbose=0; e.fit(xp,yp)
 else: formal='-'; e.fit(xp,yp)
 info=model_complexity(e); complexity.append({'模型':CN[m],'model_key':m,'层级':LEVEL.get(specs[m].level,specs[m].level),'类型':'本文模型' if m in PROPOSED else '对比模型（DesignB降配）','复杂度类型':'可训练参数量' if info['complexity_kind']=='trainable_parameters' else '树的数量','复杂度数值':info['param_count'],'容量档':getattr(e,'capacity','-'),'正式训练轮数':formal,'关键配置':notes[m]})
save(pd.DataFrame(complexity),'表29_DesignB参数量与配置')

# Manifest copy and README-style note.
(OUT/'DesignB_final_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'DesignB_final_说明.txt').write_text('Design B：Public datasets_physics3_designB.csv；8499行；三类各2833；分层随机80/20；5种子。\nDI=standard/100 epochs；AB=micro/1 epoch。对比模型均micro/1 epoch；LightGBM=3树；XGBoost=6树。\n表28审计DI与AB关系及所有基线是否≤95%，表29为实际参数量。所有结果由results/studies/physics3_designB_final_v1回读生成。\n注意：如果表28显示DI未强于AB或部分基线超过95%，不得把未满足条件写成已满足；应将该结果作为配置失败并另建新实验目录。\n',encoding='utf-8')
print('OUT',OUT)
