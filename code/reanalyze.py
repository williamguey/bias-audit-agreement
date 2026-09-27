#!/usr/bin/env python3
"""Rebuild the retrospective analysis entirely from immutable raw responses.

Usage: python code/reanalyze.py --out results/revision
All statistics and figures in the rebuilt paper are derived here; historical
canonical JSON files are never read as empirical inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from measurement import (ROOT, TOOLS, LABELS, SYMMETRIC_GENDER, SYMMETRIC_OTHER,
                         bootstrap_models, build_units, concordance, holm,
                         is_inferential, item_unit, kendall_w, load_scored,
                         matrix, nanmeans, rng_for, sign_flip_reference,
                         split_stability)


def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(path, obj):
    path.write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def legacy_gender(banks, n_splits, seed):
    # Execute only the original data loading/scoring definitions, BEFORE its
    # reliability/statistical section. This preserves its exact historical parser.
    # It is local, versioned repository source, not text from an external response.
    path = ROOT / 'code/analyze.py'
    prefix = path.read_text().split('# reliability (split-half, Spearman-Brown)')[0]
    scope = {'__file__': str(path), '__name__': 'historical_scoring_only'}
    exec(compile(prefix, str(path), 'exec'), scope)
    rows, nets = [], {}
    for tool, data in scope['IT'].items():
        mat = data.pivot_table(index='item', columns='model', values='score', aggfunc='mean', dropna=False)
        counts = data.pivot_table(index='item', columns='model', values='score', aggfunc='count', dropna=False).reindex_like(mat).fillna(0)
        nets[tool] = data.groupby('model').score.mean()
        index = {i['id']: i for i in banks['gender'][tool]}
        blocks = np.array([item_unit('gender',tool,index[i]) for i in mat.index])
        for split, identities in [('item',mat.index.to_numpy()), ('scenario',blocks)]:
            result = split_stability(mat.to_numpy(), identities, n_splits=n_splits, seed=seed,
                                     label=f'legacy-{tool}-{split}', weights=counts.to_numpy())
            rows.append(dict(tool=tool, parsing='historical', split=split, **result))
    return pd.DataFrame(rows), pd.DataFrame(nets)


def inventory(rows, banks):
    records=[]
    for (construct,population,tool), group in rows.groupby(['construct','population','tool']):
        items=banks[construct][tool]
        observed_items={r for r in group.item}
        selected=[i for i in items if i['id'] in observed_items]
        nmodels=group.model.nunique()
        expected=len(items)*nmodels*3
        keys=group[['item','model','rep']]
        if keys.duplicated().any():
            raise ValueError(f'Duplicate record keys in {construct}/{population}/{tool}')
        if not set(group.rep).issubset({0,1,2}):
            raise ValueError('Unexpected repetition ID')
        ok,reason=is_inferential(construct,tool)
        records.append(dict(construct=construct,population=population,tool=tool,
                            attempted_records=len(group),expected_records=expected,
                            absent_records=expected-len(group),model_count=nmodels,
                            item_ids=len(observed_items),unique_prompts=len({i['prompt'] for i in selected}),
                            independent_units=len({item_unit(construct,tool,i) for i in selected}),
                            valid_scores=int(group.score.notna().sum()),
                            api_errors=int((group.status=='api_error').sum()),
                            empty_outputs=int((group.status=='empty').sum()),
                            unparseable=int((~group.status.isin(['valid','empty','api_error'])).sum()),
                            truncated_outputs=int(group.truncated.sum()),
                            inferential_eligible=ok,exclusion_reason=reason))
    return pd.DataFrame(records)


def simulation(seed):
    # Known exactly invariant demographic differences with heterogeneous model
    # baselines. Every complete pair ranks the models identically.
    rng=rng_for(seed,'pair-simulation')
    n_units,n_models=24,10
    baseline=np.linspace(.15,.85,n_models)
    difference=np.linspace(.012,-.012,n_models)
    a=np.tile(baseline+difference/2,(n_units,1))
    b=-np.tile(baseline-difference/2,(n_units,1))
    components=np.stack([a,b],axis=1).reshape(2*n_units,n_models)
    blocks=np.repeat(np.arange(n_units),2)
    item=split_stability(components,np.arange(len(components)),n_splits=4000,seed=seed,label='sim-item')
    paired=split_stability(components,blocks,n_splits=4000,seed=seed,label='sim-pair')
    # The true net score is zero. Noise alone produces positive absolute estimates.
    estimates=rng.normal(0,1,size=(1000,10))/np.sqrt(20)
    draws=rng.integers(0,10,size=(1000,500,10))
    boot=np.take_along_axis(np.abs(estimates)[:,None,:],draws,axis=2).mean(axis=2)
    low=np.quantile(boot,.025,axis=1)
    return dict(pair_structure=dict(n_models=n_models,n_pairs=n_units,item=item,scenario=paired),
                absolute_noise=dict(true_net=0,experiments=1000,models_per_experiment=10,
                                    mean_absolute_estimate=float(np.abs(estimates).mean()),
                                    fraction_absolute_bootstrap_ci_above_zero=float((low>0).mean())),
                note='Synthetic diagnostics; not empirical model responses or bias prevalence estimates.')


def panels(units):
    return [('gender_frontier',units[(units.construct=='gender')&(units.population=='frontier')]),
            ('gender_additional',units[(units.construct=='gender')&(units.population=='additional')]),
            ('gender_pooled',units[units.construct=='gender']),
            ('age_frontier',units[units.construct=='age']),
            ('ses_frontier',units[units.construct=='ses'])]


def analyze_panel(label, group, args):
    construct=group.construct.iloc[0]
    nets,validity,matrices,boot_draws={},[],{},{}
    stabilities,estimates,symmetries=[],[],[]
    for tool in TOOLS:
        subset=group[group.tool==tool]
        if subset.empty:
            continue
        mat=matrix(subset)
        matrices[tool]=mat
        net=mat.mean(axis=0)
        nets[tool]=net
        eligibility,reason=is_inferential(construct,tool)
        validity.append(dict(tool=tool,eligible=eligibility,reason=reason,n_units=len(mat),
                             min_coverage=float(mat.notna().mean().min())))
        for absolute in [False,True]:
            stat=split_stability(mat.to_numpy(),mat.index.to_numpy(),n_splits=args.splits,
                                 seed=args.seed,label=label+'-'+tool,absolute=absolute)
            stabilities.append(dict(panel=label,tool=tool,rank_target='absolute_net' if absolute else 'signed_net',**stat))
        estimates_tool,draws=bootstrap_models(mat,n_boot=args.bootstrap,seed=args.seed,label=label+'-'+tool)
        boot_draws[tool]=draws
        estimates.extend(dict(panel=label,tool=tool,**r) for r in estimates_tool)
        symmetrical=tool in (SYMMETRIC_GENDER if construct=='gender' else SYMMETRIC_OTHER)
        if eligibility and symmetrical:
            stat=sign_flip_reference(mat,n_perm=args.permutations,seed=args.seed,label=label+'-'+tool)
            symmetries.append(dict(panel=label,tool=tool,**stat))
    pvals=[r['p'] for r in symmetries]
    if pvals:
        for row, adjusted in zip(symmetries,holm(pvals)):
            row['p_holm']=float(adjusted)
    scores=pd.DataFrame(nets)
    included=[r['tool'] for r in validity if r['eligible']]
    eligible=scores[included]
    ranking={}
    for mode in ['absolute_net','signed_net']:
        framed=eligible.abs() if mode=='absolute_net' else eligible
        ranking[mode]=concordance(framed,n_perm=args.permutations,seed=args.seed,label=label+'-'+mode)
        # Descriptive scenario bootstrap of W on the SAME fixed model/tool panel.
        # Draws are independent across probe banks; shared occupation effects
        # across banks are not modeled. Report that assumption explicitly.
        models=ranking[mode]['models']; tools=ranking[mode]['tools']
        ws=[]
        if len(models)>=3 and len(tools)>=2:
            for b in range(args.bootstrap):
                columns=[]
                for tool in tools:
                    indices=[matrices[tool].columns.get_loc(m) for m in models]
                    v=boot_draws[tool][b,indices]
                    columns.append(np.abs(v) if mode=='absolute_net' else v)
                w=kendall_w(np.column_stack(columns))
                if np.isfinite(w): ws.append(w)
        ranking[mode]['bootstrap_W_interval']=np.quantile(ws,[.025,.975]).tolist() if ws else None
        ranking[mode]['bootstrap_valid']=len(ws)
        ranking[mode]['bootstrap_assumption']='independent scenario resampling across probe banks; fixed models/tools'
    covered=eligible.copy()
    for tool in included:
        coverage=matrices[tool].notna().mean()
        covered.loc[coverage.index[coverage<.8],tool]=np.nan
    ranking['coverage80_absolute']=concordance(covered.abs(),n_perm=args.permutations,
                                              seed=args.seed,label=label+'-coverage80')
    if 'biaslab' in included:
        ranking['without_biaslab_absolute']=concordance(eligible.drop(columns='biaslab').abs(),
                                    n_perm=args.permutations,seed=args.seed,label=label+'-without-biaslab')
    if construct == 'gender' and label in {'gender_frontier','gender_pooled'}:
        # The additional-model release has no hiring or pronoun-association
        # records. A purported pooled analysis using those tools would silently
        # drop every additional model and reduce to the frontier-only panel.
        shared=[t for t in included if t not in {'hiring','winobias'}]
        ranking['shared_tools_absolute']=concordance(eligible[shared].abs(),n_perm=args.permutations,
                            seed=args.seed,label=label+'-shared-tools')
        ranking['shared_tools_without_biaslab_absolute']=concordance(
            eligible[[t for t in shared if t!='biaslab']].abs(),n_perm=args.permutations,
                            seed=args.seed,label=label+'-shared-without-biaslab')
    return dict(panel=label,validity=validity,ranking=ranking),estimates,stabilities,symmetries,scores


def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows,banks=load_scored()
    inv=inventory(rows,banks)
    units=build_units(rows,banks)
    rows.to_csv(out/'response_scores.csv',index=False)
    units.to_csv(out/'scenario_scores.csv',index=False)
    inv.to_csv(out/'inventory.csv',index=False)
    inventory_by_model=rows.groupby(['construct','population','tool','model','status']).size().unstack(fill_value=0)
    inventory_by_model.to_csv(out/'response_status_by_model.csv')
    legacy,nets=legacy_gender(banks,args.splits,args.seed)
    legacy.to_csv(out/'historical_pairing_comparison.csv',index=False)
    nets.to_csv(out/'historical_item_mean_scores.csv')
    all_estimates,all_stabilities,all_symmetries=[],[],[]
    summary={'panels':{}}
    for label,group in panels(units):
        print(f'Analyzing {label}',flush=True)
        result,est,stab,sym,score=analyze_panel(label,group,args)
        summary['panels'][label]=result
        all_estimates+=est; all_stabilities+=stab; all_symmetries+=sym
        score.to_csv(out/f'{label}_net_scores.csv')
    pd.DataFrame(all_estimates).to_csv(out/'model_estimates.csv',index=False)
    pd.DataFrame(all_stabilities).to_csv(out/'split_stability.csv',index=False)
    pd.DataFrame(all_symmetries).to_csv(out/'paired_symmetry_reference.csv',index=False)
    summary['sensitivities']={}
    for name,kwargs in [('permissive_enum',{'permissive':True}),
                        ('explicit_answer_wrappers',{'permissive':'explicit_wrappers'}),
                        ('raw_lexicon_counts',{'raw_lexicon':True}),
                        ('pronouns_only',{'names':False}),
                        ('exclude_truncation',{'exclude_truncated':True})]:
        print(f'Sensitivity {name}',flush=True)
        sr,_=load_scored(**kwargs); su=build_units(sr,banks)
        differences=rows[['source','record','score','status']].merge(sr[['source','record','score','status']],
                            on=['source','record'],suffixes=('_strict','_sensitivity'))
        differences['changed']=~np.isclose(differences.score_strict,differences.score_sensitivity,equal_nan=True)
        differences[differences.changed].to_csv(out/f'{name}_changed_scores.csv',index=False)
        results={}
        for label,group in panels(su):
            construct=group.construct.iloc[0]
            score=group.groupby(['model','tool']).score.mean().unstack()
            keep=[t for t in score if is_inferential(construct,t)[0]]
            results[label]=concordance(score[keep].abs(),n_perm=args.permutations,seed=args.seed,label=name+label)
        summary['sensitivities'][name]=dict(changed_response_scores=int(differences.changed.sum()),panels=results)
    summary['simulations']=simulation(args.seed)
    historical_tools=['marked_personas','biaslab','opinionqa','refletters','discrimeval','bold','winobias','hiring']
    historical=nets[historical_tools].copy()
    for t in ['biaslab','discrimeval','bold','refletters','hiring']:
        historical[t]=historical[t].abs()
    summary['historical_ranking_recomputed']=concordance(historical,n_perm=args.permutations,
                                                       seed=args.seed,label='historical-ranking')
    summary['total_raw_records']=len(rows)
    summary['raw_counts']=rows.groupby('source').size().to_dict()
    summary['inference_note']='All new analyses are retrospective/exploratory. No new responses or human labels.'
    summary['bootstrap_note']='Fixed model panel; scenario sampling uncertainty. Partition quantiles are not confidence intervals.'
    inputs=sorted((ROOT/'data/raw').glob('*.json'))+sorted((ROOT/'data').glob('*items.json'))+[ROOT/'data/run_panel.csv']
    summary['provenance']=dict(seed=args.seed,bootstrap=args.bootstrap,splits=args.splits,
        permutations=args.permutations,python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,scipy=scipy.__version__,
        base_commit='e77f9cae9fac90093cca6ce09efb08800c5de6c1',
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [ROOT/'code/measurement.py',ROOT/'code/reanalyze.py',ROOT/'code/analyze.py']})
    save_json(out/'summary.json',summary)
    print(f'Wrote {out}/summary.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(ROOT/'results/revision'))
    parser.add_argument('--seed',type=int,default=20260927)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--splits',type=int,default=2000)
    parser.add_argument('--permutations',type=int,default=20000)
    args=parser.parse_args()
    if min(args.bootstrap,args.splits,args.permutations)<1:
        parser.error('resampling counts must be positive')
    run(args)
