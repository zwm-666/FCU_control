"""Summarize the three branch-primary ablation conditions.

Reads completed per-seed records only. For each model/variant/condition,
reports mean±sample-sd accuracy, paired t and Wilcoxon p-values against that
condition's full model, and direction consistency across clean/SNR10/train20pct.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'results'/'studies'
RUNS={
 'clean':'ablation_designB_branch_v1_clean',
 'snr10':'ablation_designB_branch_v1_snr10',
 'train20pct':'ablation_designB_branch_v1_train20pct',
}

def load(name): return json.loads((S/name/'per_seed_records.json').read_text(encoding='utf-8'))
def paired(a,b):
 seeds=sorted(set(a)&set(b)); x=np.array([a[s] for s in seeds]); y=np.array([b[s] for s in seeds]); d=(x-y)*100
 return {'delta_pp':[round(float(v),4) for v in d],'mean_pp':round(float(d.mean()),4),'sd_pp':round(float(d.std(ddof=1)),4),'t_p':float(stats.ttest_rel(x,y).pvalue),'wilcoxon_p':float(stats.wilcoxon(x,y).pvalue),'wins_variant_better':f'{int((d>0).sum())}/{len(d)}'}

def main():
 allr={k:load(v) for k,v in RUNS.items()}
 summaries=[]
 for cond,recs in allr.items():
  groups={}
  for r in recs: groups.setdefault((r['model'],r['variant']),{})[int(r['seed'])]=float(r['accuracy'])
  for model in ('di_emstgat','ab_emstgat'):
   full=groups[(model,'full')]
   for (m,var), vals in sorted(groups.items()):
    if m!=model or var=='full': continue
    t=paired(vals,full); t.update({'condition':cond,'model':model,'variant':var,'full_mean_pct':float(np.mean(list(full.values()))*100),'variant_mean_pct':float(np.mean(list(vals.values()))*100)}); summaries.append(t)
  for model in ('di_emstgat','ab_emstgat'):
   f=groups[(model,'full')]; print(f'{cond:10s} {model:12s} full={np.mean(list(f.values()))*100:.4f}% sd={np.std(list(f.values()),ddof=1)*100:.4f}')
 print('\n=== AB component directions (delta=removed-full; + means removal improves) ===')
 for var in sorted({r['variant'] for r in summaries if r['model']=='ab_emstgat'}):
  x=[r for r in summaries if r['model']=='ab_emstgat' and r['variant']==var]
  print(var, ' / '.join(f"{r['condition']} {r['mean_pp']:+.4f} p={r['t_p']:.4f} {r['wins_variant_better']}" for r in x), 'signs', [np.sign(r['mean_pp']) for r in x])
 out={'runs':RUNS,'records':{k:len(v) for k,v in allr.items()},'summaries':summaries,'interpretation':'Exploratory; AB primary compact/100, DI comparison light/1; no component necessity claim unless direction consistent across all three conditions.'}
 dest=S/'ablation_designB_branch_v1_statistics.json'; dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print('Saved:',dest)
if __name__=='__main__': main()
