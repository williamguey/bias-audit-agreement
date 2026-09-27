#!/usr/bin/env python3
"""Generate publication figures, tables, and LaTeX numbers from reanalysis files."""
import json
import os
import re
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from measurement import ROOT,TOOLS,LABELS

RESULTS=ROOT/'results/revision'
FIGURES=ROOT/'paper/figures'
TABLES=ROOT/'paper/generated'
FIGURES.mkdir(parents=True,exist_ok=True);TABLES.mkdir(parents=True,exist_ok=True)
summary=json.loads((RESULTS/'summary.json').read_text())
inventory=pd.read_csv(RESULTS/'inventory.csv')
pairing=pd.read_csv(RESULTS/'historical_pairing_comparison.csv')
stability=pd.read_csv(RESULTS/'split_stability.csv')
estimates=pd.read_csv(RESULTS/'model_estimates.csv')
symmetry=pd.read_csv(RESULTS/'paired_symmetry_reference.csv')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
BLUE='#17648c';ORANGE='#c85d2c';GRAY='#68747c'


def save(fig,name):
    fig.savefig(FIGURES/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIGURES/f'{name}.png',bbox_inches='tight',dpi=220)
    plt.close(fig)


def fmt(v,d=3):
    return '--' if v is None or not np.isfinite(v) else f'{v:.{d}f}'


def pformat(p):
    return '--' if p is None else ('<0.001' if p<.001 else f'{p:.3f}')


def table(name,header,rows,align):
    text='\\begin{tabular}{'+align+'}\n\\toprule\n'+ ' & '.join(header)+' \\\\\n\\midrule\n'
    for row in rows:text+=' & '.join(re.sub(r'(?<!\\)%',r'\\%',str(x)) for x in row)+' \\\\\n'
    text+='\\bottomrule\n\\end{tabular}\n'
    (TABLES/f'{name}.tex').write_text(text)


# Pairing intervention under identical historical scoring and response weights.
paired_tools=['biaslab','refletters','discrimeval','bold','winobias']
fig,ax=plt.subplots(figsize=(6.8,3.45),layout='constrained')
for i,t in enumerate(paired_tools):
    row=pairing[pairing.tool==t].set_index('split')
    a,b=row.loc['item','mean_r'],row.loc['scenario','mean_r']
    ax.plot([a,b],[i,i],color='#c7cdd2',lw=3)
    ax.scatter(a,i,color=ORANGE,s=42,label='Individual-item split' if i==0 else None,zorder=3)
    ax.scatter(b,i,color=BLUE,s=42,label='Complete-scenario split' if i==0 else None,zorder=3)
    ax.text(a-.035,i-.18,f'{a:.2f}',ha='right',fontsize=8,color=ORANGE)
    ax.text(b+.035,i+.17,f'{b:.2f}',ha='left',fontsize=8,color=BLUE)
ax.set_yticks(range(len(paired_tools)),[LABELS[t] for t in paired_tools]);ax.invert_yaxis()
ax.axvline(0,color=GRAY,lw=.8,ls='--');ax.set_xlim(-.68,.9)
ax.set_xlabel('Mean split-half Spearman correlation (signed scores)')
ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=2,frameon=False,fontsize=8)
ax.grid(axis='x',alpha=.15)
save(fig,'pairing')

primary=summary['panels']['gender_frontier']['ranking']
specs=[('Historical implementation',summary['historical_ranking_recomputed']),
       ('Revised: absolute net',primary['absolute_net']),
       ('Revised: signed net',primary['signed_net']),
       ('Without mirrored assertions',primary['without_biaslab_absolute']),
       ('Explicit first-answer parser',summary['sensitivities']['permissive_enum']['panels']['gender_frontier']),
       ('Explicit answer wrappers',summary['sensitivities']['explicit_answer_wrappers']['panels']['gender_frontier']),
       ('Raw lexical counts',summary['sensitivities']['raw_lexicon_counts']['panels']['gender_frontier']),
       ('Exclude truncated outputs',summary['sensitivities']['exclude_truncation']['panels']['gender_frontier']),
       ('At least 80% unit coverage',primary['coverage80_absolute'])]
fig,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained')
for i,(name,r) in enumerate(specs):
    ax.scatter(r['W'],i,s=42,color=ORANGE if i==0 else BLUE)
    ax.text(.39,i,f"n={r['n_models']}, k={r['n_tools']}; p={pformat(r['permutation_p'])}",va='center',fontsize=8)
ax.set_yticks(range(len(specs)),[name for name,r in specs]);ax.invert_yaxis()
ax.set_xlim(0,.70);ax.set_xticks([0,.1,.2,.3]);ax.set_xlabel("Kendall's W")
ax.grid(axis='x',alpha=.2)
save(fig,'specifications')

# Unit coverage measures complete usable scenarios, not mere API response rates.
cover=estimates[estimates.panel=='gender_frontier'].pivot(index='model',columns='tool',values='unit_coverage')
cover=cover.reindex(columns=TOOLS)
fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
im=ax.imshow(100*cover.to_numpy(),vmin=0,vmax=100,cmap='Blues',aspect='auto')
for i in range(len(cover)):
    for j in range(len(cover.columns)):
        v=100*cover.iloc[i,j]
        ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=7,color='white' if v>60 else '#25333c')
ax.set_xticks(range(len(TOOLS)),[LABELS[t] for t in TOOLS],rotation=40,ha='right')
ax.set_yticks(range(len(cover)),cover.index);ax.tick_params(length=0)
fig.colorbar(im,ax=ax,label='Usable complete scenarios (%)',fraction=.03,pad=.02)
save(fig,'coverage')

fig,axes=plt.subplots(2,2,figsize=(7.1,6.0),layout='constrained')
rank_tools=['marked_personas','biaslab','winobias','hiring']
for ax,tool in zip(axes.ravel(),rank_tools):
    e=estimates[(estimates.panel=='gender_frontier')&(estimates.tool==tool)].copy()
    e=e[e.net.notna()].sort_values('net',key=lambda x:x.abs()).reset_index(drop=True)
    scores=e.net.abs().rank(method='average').to_numpy()
    y=np.arange(len(e))
    for i in y:ax.plot([e.loc[i,'rank_ci_low'],e.loc[i,'rank_ci_high']],[i,i],color=BLUE,lw=1.5)
    ax.scatter(scores,y,color=BLUE,s=14,zorder=3)
    ax.set_yticks(y,e.model,fontsize=7.5);ax.invert_yaxis()
    ax.set_title(LABELS[tool],loc='left',fontweight='bold');ax.set_xlim(.5,10.5)
    ax.set_xticks([1,3,5,7,9]);ax.set_xlabel('Rank of absolute net score',fontsize=8)
    ax.grid(axis='x',alpha=.15)
save(fig,'rank_uncertainty')

gender=inventory[(inventory.construct=='gender')&(inventory.population=='frontier')].set_index('tool')
rows=[]
for t in TOOLS:
    r=gender.loc[t]
    rows.append([LABELS[t],r.item_ids,r.independent_units,r.attempted_records,
                 f'{100*r.valid_scores/r.attempted_records:.1f}\\%',
                 'Descriptive' if t=='opinionqa' else 'Exploratory'])
table('inventory',['Probe family','IDs','Units','Records','Parsed','Use'],rows,'lrrrrl')

rows=[]
for t in paired_tools:
    d=pairing[pairing.tool==t].set_index('split')
    rows.append([LABELS[t],fmt(d.loc['item','mean_r']),fmt(d.loc['scenario','mean_r']),
                 f"[{d.loc['scenario','partition_q025']:.2f}, {d.loc['scenario','partition_q975']:.2f}]"])
table('pairing',['Probe family','Item split','Scenario split','Scenario partition range'],rows,'lrrr')

rows=[]
for name,r in specs:rows.append([name,r['n_models'],r['n_tools'],fmt(r['W']),pformat(r['permutation_p'])])
table('concordance',['Specification','Models','Probes','$W$','$p$'],rows,'lrrrr')

rows=[]
for r in symmetry[symmetry.panel=='gender_frontier'].itertuples():
    rows.append([LABELS[r.tool],fmt(r.observed),fmt(r.null_q95),pformat(r.p),pformat(r.p_holm)])
table('symmetry',['Paired probe','Observed $D$','Null 95th pct.','$p$','Holm $p$'],rows,'lrrrr')

rows=[]
extension=[('Additional panel, shared probes','gender_additional','absolute_net'),
           ('Original panel, shared probes','gender_frontier','shared_tools_absolute'),
           ('Expanded panel, shared probes','gender_pooled','shared_tools_absolute'),
           ('Expanded, without assertions','gender_pooled','shared_tools_without_biaslab_absolute'),
           ('Age, eligible probes','age_frontier','absolute_net'),
           ('SES, eligible probes','ses_frontier','absolute_net')]
for name,panel,spec in extension:
    r=summary['panels'][panel]['ranking'][spec]
    rows.append([name,r['n_models'],r['n_tools'],fmt(r['W']),pformat(r['permutation_p'])])
table('extensions',['Analysis','Models','Probes','$W$','$p$'],rows,'lrrrr')

rows=[]
for t in TOOLS:
    d=stability[(stability.panel=='gender_frontier')&(stability.tool==t)].set_index('rank_target')
    rows.append([LABELS[t],fmt(d.loc['signed_net','mean_r']),fmt(d.loc['absolute_net','mean_r']),
                 f"{int(d.loc['signed_net','valid_splits'])}/{int(d.loc['signed_net','total_splits'])}"])
table('revised_stability',['Probe','Signed rank $r$','Absolute rank $r$','Defined signed splits'],rows,'lrrr')

numbers={'TotalRecords':f"{summary['total_raw_records']:,}",
         'MainW':fmt(primary['absolute_net']['W']),'MainP':pformat(primary['absolute_net']['permutation_p']),
         'SignedW':fmt(primary['signed_net']['W']),'SignedP':pformat(primary['signed_net']['permutation_p']),
         'HistoricalW':fmt(summary['historical_ranking_recomputed']['W']),
         'HistoricalP':pformat(summary['historical_ranking_recomputed']['permutation_p']),
         'WithoutW':fmt(primary['without_biaslab_absolute']['W']),
         'WithoutP':pformat(primary['without_biaslab_absolute']['permutation_p']),
         'PermissiveW':fmt(summary['sensitivities']['permissive_enum']['panels']['gender_frontier']['W']),
         'PermissiveP':pformat(summary['sensitivities']['permissive_enum']['panels']['gender_frontier']['permutation_p']),
         'BootstrapN':str(summary['provenance']['bootstrap']),
         'SplitN':str(summary['provenance']['splits']),
         'PermutationN':f"{summary['provenance']['permutations']:,}",
         'SimItemR':fmt(summary['simulations']['pair_structure']['item']['mean_r']),
         'SimPairR':fmt(summary['simulations']['pair_structure']['scenario']['mean_r'])}
for t,abbr in [('discrimeval','Decision'),('bold','Continuation'),('winobias','Pronoun')]:
    d=pairing[pairing.tool==t].set_index('split')
    numbers[abbr+'ItemR']=fmt(d.loc['item','mean_r'])
    numbers[abbr+'PairR']=fmt(d.loc['scenario','mean_r'])
(TABLES/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in numbers.items())+'\n')
print('Generated four figures, six tables, and result macros.')
