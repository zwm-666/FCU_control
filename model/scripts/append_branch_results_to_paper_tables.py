"""Append branch-primary results to the existing thesis paper tables.

This is additive only: existing rows are read, new rows receive distinct
record-type names and provenance, and the files are rewritten with the union of
columns. No old row is deleted or edited.

Current paper output layout (after build_paper_tables.py):
  论文表1_不同模型故障诊断性能对比.csv
  论文表2_不同模型噪声鲁棒性分析.csv
  论文表3_分支结构消融实验.csv
"""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path
from collections import Counter

ROOT=Path(r"D:/my_project/h2-fcu-modern-dashboard/model")
RESULTS=Path(r"D:/learn/毕业材料/graduation/results")
STUDIES=ROOT/'results'/'studies'
BACKUP=RESULTS/'表格追加前备份_20260831'

TABLE1=RESULTS/'论文表1_不同模型故障诊断性能对比.csv'
TABLE2=RESULTS/'论文表2_不同模型噪声鲁棒性分析.csv'
TABLE3=RESULTS/'论文表3_分支结构消融实验.csv'
MAIN=STUDIES/'physics3_designB_branch_detuned_v1'/'aggregate_summary.csv'
NOISE=STUDIES/'noise_designB_branch_v1'/'aggregate_summary.csv'
ABL={k:STUDIES/v/'aggregate_summary.csv' for k,v in {
 'clean':'ablation_designB_branch_v1_clean',
 'snr10':'ablation_designB_branch_v1_snr10',
 'train20pct':'ablation_designB_branch_v1_train20pct',
}.items()}


def read(p):
 with p.open(encoding='utf-8-sig',newline='') as h:
  r=csv.DictReader(h); return list(r),list(r.fieldnames or [])

def write(p,rows,fields):
 with p.open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def add_rows(table, source, block, provenance, transform):
 rows,fields=read(table); src,src_fields=read(source)
 new=[]
 for r in src:
  x={k:r.get(k,'') for k in src_fields}; x=transform(x)
  x['记录类型']=block; x['来源分项表']=provenance
  new.append(x)
 all_fields=[]
 for f in fields+list(new[0].keys()):
  if f not in all_fields: all_fields.append(f)
 write(table,rows+new,all_fields)
 return len(rows),len(new),len(rows)+len(new)

def pct(v):
 try:return f'{float(v)*100:.4f}'
 except:return v

def main():
 BACKUP.mkdir(exist_ok=True)
 for p in (TABLE1,TABLE2,TABLE3): shutil.copy2(p,BACKUP/p.name)
 before={p.name:read(p)[0] for p in (TABLE1,TABLE2,TABLE3)}
 results=[]
 # Main summary: source aggregate uses decimal metrics.
 n,add,total=add_rows(TABLE1,MAIN,'分支主模型排名_降配DI_compactAB100','physics3_designB_branch_detuned_v1/aggregate_summary.csv',lambda r:{
  '提纲小节':'4.4.5','模型':{'ab_emstgat':'AB-EMSTGAT（本文）','di_emstgat':'DI-EMSTGAT（降配对照）'}.get(r.get('model'),r.get('model')),
  'model_key':r.get('model'),'类型':'本文' if r.get('model')=='ab_emstgat' else '对比',
  'Accuracy均值(%)':pct(r.get('accuracy_mean')),'Accuracy标准差(pp)':pct(r.get('accuracy_std')),
  'MacroF1均值(%)':pct(r.get('f1_macro_mean')),'MacroF1标准差(pp)':pct(r.get('f1_macro_std')),
  'Kappa均值(%)':pct(r.get('cohen_kappa_mean')),'训练耗时均值(s)':r.get('fit_time_mean',''),
  '参数量':r.get('param_count',''),'运行次数':r.get('n_seeds',''),
  '层级':'branch_variant_primary',
 }); results.append(('table1',n,add,total))
 # Noise summary aggregate has model/snr rows; make compact body as one block.
 n,add,total=add_rows(TABLE2,NOISE,'分支主模型噪声_全网格汇总','noise_designB_branch_v1/aggregate_summary.csv',lambda r:{
  '提纲小节':'4.4.6','模型':{'ab_emstgat':'AB-EMSTGAT（本文）','di_emstgat':'DI-EMSTGAT（降配对照）'}.get(r.get('model'),r.get('model')),
  'model_key':r.get('model'),'种子数':r.get('n_seeds',''),'SNR(dB)':r.get('snr_db',''),
  'Accuracy_均值(%)':pct(r.get('accuracy_mean')),'Accuracy_标准差(%)':pct(r.get('accuracy_std')),
  'MacroF1_均值(%)':pct(r.get('f1_macro_mean')),'MacroF1_标准差(%)':pct(r.get('f1_macro_std')),
  '层级':'branch_variant_primary',
 }); results.append(('table2',n,add,total))
 # Ablation: retain every aggregate row in one new table block per condition.
 for cond,src in ABL.items():
  n,add,total=add_rows(TABLE3,src,f'分支主模型消融_{cond}','/'.join(src.parts[-3:]),lambda r,cond=cond:{
   '提纲小节':'4.4.3','条件':{'clean':'干净条件','snr10':'SNR=10dB','train20pct':'训练集20%' }[cond],
   '模型':{'ab_emstgat':'AB-EMSTGAT（本文）','di_emstgat':'DI-EMSTGAT（降配对照）'}.get(r.get('model'),r.get('model')),
   'model_key':r.get('model'),'变体':r.get('variant_cn',''),'variant_key':r.get('variant',''),
   '参数量':r.get('param_count',''),'运行次数':r.get('n_seeds',''),
   '完整模型Accuracy(%)':pct(float(r.get('accuracy_mean','0')) - float(r.get('delta_accuracy_points','0')) / 100.0),
   '变体Accuracy(%)':pct(r.get('accuracy_mean')),
   '变体Accuracy标准差(pp)':pct(r.get('accuracy_std')),
   '变化(pp)':r.get('delta_accuracy_points',''),'配对t检验p':r.get('paired_t_p',''),
   'Wilcoxon p':r.get('wilcoxon_p',''),'变体更差种子':r.get('variant_worse_seeds',''),
   '对比状态':'branch_variant',
  }); results.append((f'table3_{cond}',n,add,total))
 # Assert old row sequences survive exactly as dicts; only append was allowed.
 for p in (TABLE1,TABLE2,TABLE3):
  now,now_fields=read(p); old,old_fields=read(BACKUP/p.name)
  if now[:len(old)] != old:
   # New union columns make dict equality too strict; compare every original
   # field/value pair and ignore only newly introduced columns.
   for i,(new_row,old_row) in enumerate(zip(now,old)):
    if any(new_row.get(f,'') != old_row.get(f,'') for f in old_fields):
     raise AssertionError(f'old row changed: {p} row {i}')
 manifest={'backup_dir':str(BACKUP),'tables':results,'deleted':[],'old_rows_preserved':True,'variant':'DesignB_branch_detuned_v1','noise_variant':'noise_designB_branch_v1','ablation_variants':list(ABL)}
 (RESULTS/'分支主模型结果追加记录_20260831.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
