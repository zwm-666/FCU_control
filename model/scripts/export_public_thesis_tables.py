import ast
import json
from pathlib import Path
import pandas as pd

src = Path('results/studies/public_10model_5seed_v1')
out = Path(r'D:/learn/毕业材料/graduation/results')
out.mkdir(parents=True, exist_ok=True)
cn = {'xgboost':'XGBoost','lightgbm':'LightGBM','tcn':'TCN','transformer':'Transformer','mtgnn':'MTGNN','gatv2':'GATv2','itransformer':'iTransformer','patchtst':'PatchTST','di_emstgat':'DI-EMSTGAT（本文）','ab_emstgat':'AB-EMSTGAT（本文）'}
lvl = {'traditional_baseline':'传统机器学习','deep_baseline':'深度序列基线','graph_spatiotemporal_baseline':'图/时空基线','modern_transformer_baseline':'现代Transformer基线','proposed_direct_input':'本文-直接输入','proposed_adaptive_branch':'本文-自动分支'}
rec = pd.read_csv(src/'per_seed_records.csv')
man = json.loads((src/'manifest.json').read_text(encoding='utf-8'))
pct = lambda x: round(float(x)*100,4)
def save(df,name):
    df.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    print(name,len(df))
rows=[]
for model,g in rec.groupby('model',sort=False):
    r={'模型':cn[model],'model_key':model,'层级':lvl.get(g.level.iloc[0],g.level.iloc[0]),'运行次数':len(g)}
    for metric,label in [('accuracy','Accuracy'),('balanced_accuracy','BalancedAcc'),('precision_weighted','Precision'),('recall_weighted','Recall'),('f1_macro','MacroF1'),('f1_weighted','WeightedF1'),('cohen_kappa','Kappa')]:
        r[f'{label}_均值(%)']=pct(g[metric].mean()); r[f'{label}_标准差(%)']=pct(g[metric].std(ddof=1))
    r['训练耗时均值(s)']=round(g.fit_time.mean(),4); r['推理耗时均值(s)']=round(g.prediction_time.mean(),4)
    rows.append(r)
save(pd.DataFrame(rows).sort_values('Accuracy_均值(%)',ascending=False),'表9_公开数据集主表_10模型5种子')
scalar=['model','level','seed','accuracy','balanced_accuracy','cohen_kappa','precision_weighted','recall_weighted','f1_weighted','f1_macro','fit_time','prediction_time','feature_count','train_samples','test_samples']
d=rec[scalar].copy(); d.insert(0,'模型',d.model.map(cn)); d.insert(1,'层级中文',d.level.map(lambda x:lvl.get(x,x)))
for c in ['accuracy','balanced_accuracy','cohen_kappa','precision_weighted','recall_weighted','f1_weighted','f1_macro']: d[c]=d[c].map(pct)
save(d,'表10_公开数据集逐种子指标')
class_rows=[]; cm_rows=[]
for _,r in rec.iterrows():
    report=ast.literal_eval(r.classification_report); cm=ast.literal_eval(r.confusion_matrix)
    for cls,v in report.items():
        if cls in {'accuracy','macro avg','weighted avg'}: continue
        class_rows.append({'模型':cn[r.model],'model_key':r.model,'seed':int(r.seed),'类别':cls,'Precision(%)':pct(v['precision']),'Recall(%)':pct(v['recall']),'F1(%)':pct(v['f1-score']),'支持数':int(v['support'])})
    cm_rows.append({'模型':cn[r.model],'model_key':r.model,'seed':int(r.seed),'混淆矩阵':json.dumps(cm,ensure_ascii=False)})
classes=pd.DataFrame(class_rows); save(classes,'表12_公开数据集逐种子类别级指标'); save(pd.DataFrame(cm_rows),'表13_公开数据集混淆矩阵')
agg=[]
for (model,cls),g in classes.groupby(['model_key','类别'],sort=False):
    agg.append({'模型':cn[model],'类别':cls,'Precision均值(%)':round(g['Precision(%)'].mean(),4),'Precision标准差(%)':round(g['Precision(%)'].std(ddof=1),4),'Recall均值(%)':round(g['Recall(%)'].mean(),4),'Recall标准差(%)':round(g['Recall(%)'].std(ddof=1),4),'F1均值(%)':round(g['F1(%)'].mean(),4),'F1标准差(%)':round(g['F1(%)'].std(ddof=1),4)})
save(pd.DataFrame(agg),'表11_公开数据集类别级指标')
config=[]
for model,g in rec.groupby('model',sort=False):
    config.append({'模型':cn[model],'层级':lvl.get(g.level.iloc[0],g.level.iloc[0]),'结果目录':'public_10model_5seed_v1','训练轮数':man['deep_epochs'],'容量策略':man.get('baseline_capacity') if model not in {'di_emstgat','ab_emstgat'} else man.get('proposed_capacity'),'特征选择':bool(g.feature_selection_used.iloc[0]),'特征数范围':f"{int(g.feature_count.min())}-{int(g.feature_count.max())}",'训练样本':int(g.train_samples.iloc[0]),'测试样本':int(g.test_samples.iloc[0]),'类别数':4,'类别':'Flooding; Membrane_Drying; Normal; Thermal_Management_Fault'})
save(pd.DataFrame(config),'表14_公开数据集实验配置')
note=f"数据文件: Public datasets.csv\n协议: {man['split_protocol']}\n测试比例: {man['test_size']}；种子: {man['seeds']}；每模型运行: 5\n类别: Flooding, Membrane_Drying, Normal, Thermal_Management_Fault\n训练/测试样本: 7076/1770；特征选择在训练集内完成，特征数随种子为8或9。\n来源: results/studies/public_10model_5seed_v1（已有完成结果，未冒充最新降配配置）。\n"
(out/'表9-14_公开数据集_说明.txt').write_text(note,encoding='utf-8')
print('OUT',out)
