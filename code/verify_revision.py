#!/usr/bin/env python3
"""Fail closed if revised outputs are missing, stale, or internally inconsistent."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from measurement import ROOT,is_inferential,kendall_w,load_banks,item_unit


def main():
    folder=ROOT/'results/revision'
    summary=json.loads((folder/'summary.json').read_text())
    for group in ['input_sha256','code_sha256']:
        for path,want in summary['provenance'][group].items():
            got=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            if got!=want:raise AssertionError(f'Stale or changed input/code: {path}; rerun reanalysis')
    responses=pd.read_csv(folder/'response_scores.csv')
    units=pd.read_csv(folder/'scenario_scores.csv')
    inventory=pd.read_csv(folder/'inventory.csv')
    estimates=pd.read_csv(folder/'model_estimates.csv')
    assert len(responses)==summary['total_raw_records']==19500
    assert responses[['source','record']].duplicated().sum()==0
    assert responses.score.notna().equals(responses.status.eq('valid'))
    assert int(inventory.attempted_records.sum())==len(responses)
    assert int(inventory.valid_scores.sum())==int(responses.score.notna().sum())
    assert int(inventory.absent_records.sum())==156
    for construct in ['age','ses']:
        row=inventory.query('construct==@construct and tool=="bbq"').iloc[0]
        assert row.item_ids==12 and row.unique_prompts==1 and row.independent_units==1
    assert (units.loc[units.complete_repetitions==0,'score'].isna()).all()
    assert (units.complete_repetitions<=3).all()
    key=['construct','population','tool','model','unit']
    assert not units[key].duplicated().any()
    banks=load_banks()
    for (construct,population,tool,model),part in units.groupby(key[:-1]):
        expected={item_unit(construct,tool,i) for i in banks[construct][tool]}
        assert set(part.unit)==expected,(construct,population,tool,model)
    for panel,data in summary['panels'].items():
        scores=pd.read_csv(folder/f'{panel}_net_scores.csv',index_col=0)
        subset=estimates[estimates.panel==panel]
        piv=subset.pivot(index='model',columns='tool',values='net').reindex_like(scores)
        np.testing.assert_allclose(scores.to_numpy(),piv.to_numpy(),equal_nan=True,atol=1e-12)
        construct='gender' if panel.startswith('gender') else panel.split('_')[0]
        for spec,result in data['ranking'].items():
            for tool in result['tools']:assert is_inferential(construct,tool)[0]
            if result['W'] is None:continue
            selected=scores.loc[result['models'],result['tools']]
            if spec!='signed_net':selected=selected.abs()
            got=kendall_w(selected.to_numpy())
            assert np.isclose(got,result['W']), (panel,spec,got,result['W'])
            # An independent SciPy Friedman implementation yields the same W.
            a=selected.to_numpy();n,k=a.shape
            oracle=friedmanchisquare(*[a[i] for i in range(n)]).statistic/(k*(n-1))
            assert np.isclose(got,oracle), (panel,spec,got,oracle)
            assert 0<result['permutation_p']<=1
    for panel,group in estimates.groupby('panel'):
        for tool,part in group.groupby('tool'):
            # Average bottom-rank shares across resamples sum to one when every
            # column has a defined conditional ranking in each draw. Allow a
            # small deficit/overshoot only when missing bootstrap columns exist.
            if (part.valid_bootstraps==summary['provenance']['bootstrap']).all():
                assert np.isclose(part.bottom_rank_share.sum(),1.),(panel,tool)
            assert part.unit_coverage.between(0,1).all()
            observed=part[part.net.notna()]
            assert (observed.net_ci_low<=observed.net_ci_high).all()
    sim=summary['simulations']
    assert sim['pair_structure']['item']['mean_r']<0
    assert np.isclose(sim['pair_structure']['scenario']['mean_r'],1)
    assert sim['absolute_noise']['fraction_absolute_bootstrap_ci_above_zero']==1
    manuscript=(ROOT/'paper/main.tex').read_text()
    bib=(ROOT/'paper/references.bib').read_text()
    available=set(re.findall(r'@\w+\{([^,]+),',bib))
    cited={k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',manuscript) for k in group.split(',')}
    assert not cited-available, f'Missing bibliography entries: {cited-available}'
    assert 'preregistration' in manuscript and 'retrospective' in manuscript
    assert 'Generative AI assistance' in manuscript
    numbers=(ROOT/'paper/generated/numbers.tex').read_text()
    assert f"\\newcommand{{\\MainW}}{{{summary['panels']['gender_frontier']['ranking']['absolute_net']['W']:.3f}}}" in numbers
    print('PASS: immutable input hashes, analysis code hashes, 19,500 records, coverage,')
    print('scenario completeness, all main W values against independent SciPy formula,')
    print('rank probabilities, synthetic mechanisms, references, and manuscript result macros.')


if __name__=='__main__':main()
