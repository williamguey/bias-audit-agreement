import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from measurement import (bootstrap_models,build_units,concordance,enum_answer,holm,
                         item_unit,kendall_w,lexical_gender,score_answer,
                         sign_flip_reference,split_stability)


class ParsingTests(unittest.TestCase):
    def test_refusals_and_discussion_are_not_choices(self):
        for response in ['As an AI, I cannot choose.','Both are qualified.','A or B',
                         'I cannot answer yes or no.','There is no reason to prefer A.']:
            self.assertIsNone(enum_answer(response,['a','b']))
            self.assertIsNone(enum_answer(response,['yes','no']))

    def test_punctuation_and_case(self):
        self.assertEqual(enum_answer(' **Strongly disagree.** ',['strongly disagree','agree']), 'strongly disagree')
        self.assertEqual(enum_answer("'A'",['a','b']),'a')
        self.assertIsNone(enum_answer('A. More explanation',['a','b']))
        self.assertEqual(enum_answer('A. More explanation',['a','b'],True),'a')

    def test_unrecognized_is_missing_not_neutral(self):
        score,status=score_answer('gender','discrimeval',{'gender':'m'},'It depends.')
        self.assertTrue(np.isnan(score))
        self.assertEqual(status,'non_enum_or_ambiguous')

    def test_explicit_wrappers_without_substring_guessing(self):
        choices=['strongly disagree','disagree','neutral','agree','strongly agree']
        for text,want in [("I'm neutral.",'neutral'),('I strongly disagree.','strongly disagree'),
                          ("I would respond with: 'neutral'",'neutral')]:
            self.assertEqual(enum_answer(text,choices,'explicit_wrappers'),want)
        self.assertIsNone(enum_answer('I do not agree.',choices,'explicit_wrappers'))
        self.assertIsNone(enum_answer('I might agree or disagree.',choices,'explicit_wrappers'))

    def test_gender_word_boundaries_and_title(self):
        self.assertEqual(lexical_gender('She is a female surgeon.'),'f')
        self.assertIsNone(lexical_gender('Dr. Taylor is a surgeon.'))
        self.assertEqual(lexical_gender('Dr. Sarah is a surgeon.'),'f')

    def test_question_polarity_for_age_ambiguity(self):
        item={'stereo_dir':'y','g_stereo':'younger'}
        self.assertEqual(score_answer('age','bbq',item,'Older')[0],1)
        self.assertEqual(score_answer('age','bbq',item,'younger')[0],-1)
        self.assertEqual(score_answer('age','bbq',item,'unknown')[0],0)

    def test_lexicon_scale_invariant_to_repeated_text(self):
        item={'gender':'m'}
        one=score_answer('gender','refletters',item,'confident and helpful but analytical')[0]
        two=score_answer('gender','refletters',item,('confident and helpful but analytical ')*2)[0]
        self.assertAlmostEqual(one,two)


class UnitTests(unittest.TestCase):
    def test_pair_identity_and_prompt_duplicates(self):
        a={'id':'disc_0_m','prompt':'A'}; b={'id':'disc_0_f','prompt':'B'}
        self.assertEqual(item_unit('gender','discrimeval',a),item_unit('gender','discrimeval',b))
        a={'id':'one','prompt':'same'}; b={'id':'two','prompt':'same'}
        self.assertEqual(item_unit('age','bbq',a),item_unit('age','bbq',b))

    def test_incomplete_pairs_and_rep_averaging(self):
        banks={'gender':{'discrimeval':[{'id':'disc_0_m','prompt':'m'},{'id':'disc_0_f','prompt':'f'}]}}
        rows=[]
        # One complete repetition: (+1,-1) = 0. Missing the female response in
        # repetition 1 must not manufacture a positive contrast from male +1.
        for rep,item,score in [(0,'disc_0_m',1.),(0,'disc_0_f',-1.),(1,'disc_0_m',1.),(1,'disc_0_f',np.nan)]:
            rows.append(dict(construct='gender',population='frontier',tool='discrimeval',model='M',
                             unit='disc_0',item=item,rep=rep,score=score))
        units=build_units(pd.DataFrame(rows),banks)
        self.assertEqual(units.iloc[0].score,0.)
        self.assertEqual(units.iloc[0].complete_repetitions,1)

    def test_completely_missing_pair_stays_missing(self):
        banks={'gender':{'discrimeval':[{'id':'disc_0_m','prompt':'m'},{'id':'disc_0_f','prompt':'f'}]}}
        rows=pd.DataFrame([dict(construct='gender',population='frontier',tool='discrimeval',model='M',
                               unit='disc_0',item='disc_0_m',rep=0,score=1.)])
        self.assertTrue(np.isnan(build_units(rows,banks).iloc[0].score))


class CoverageTests(unittest.TestCase):
    def test_bank_scenario_absent_for_all_models_stays_in_denominator(self):
        items=[{'id':f'disc_{i}_{g}','prompt':f'{i}-{g}'} for i in range(2) for g in ['m','f']]
        banks={'gender':{'discrimeval':items}}
        rows=pd.DataFrame([dict(construct='gender',population='additional',tool='discrimeval',
                                model=model,unit='disc_0',item=f'disc_0_{g}',rep=0,score=score)
                           for model in ['M','N'] for g,score in [('m',1.),('f',-1.)]])
        units=build_units(rows,banks)
        self.assertEqual(len(units),4)
        self.assertTrue(units[units.unit=='disc_1'].score.isna().all())
        coverage=units.groupby('model').score.apply(lambda x:x.notna().mean())
        np.testing.assert_allclose(coverage.to_numpy(),[.5,.5])


class StatisticsTests(unittest.TestCase):
    def test_tie_corrected_concordance(self):
        a=np.array([[1,2],[1,2],[3,4],[4,5.]])
        self.assertAlmostEqual(kendall_w(a),1.)
        reverse=np.array([[1,3],[2,2],[3,1.]])
        self.assertAlmostEqual(kendall_w(reverse),0.)
        self.assertTrue(np.isnan(kendall_w(np.ones((4,2)))))

    def test_permutation_p_never_zero_and_constant_exclusion(self):
        frame=pd.DataFrame({'a':range(6),'b':range(6),'flat':[1]*6})
        result=concordance(frame,n_perm=99,seed=3)
        self.assertGreaterEqual(result['permutation_p'],.01)
        self.assertEqual(result['constant_tools'],['flat'])

    def test_sign_flip_exact_and_joint(self):
        a=np.array([[1.,2.],[1.,2.],[1.,2.]])
        result=sign_flip_reference(a)
        self.assertTrue(result['exact'])
        self.assertEqual(result['patterns'],8)
        self.assertAlmostEqual(result['p'],2/8)
        self.assertEqual(sign_flip_reference(np.zeros((3,2)))['p'],1.)

    def test_holm(self):
        np.testing.assert_allclose(holm([.04,.01,.03]),[.06,.03,.06])

    def test_block_split_known_perfect_stability(self):
        baseline=np.linspace(.1,.9,10); delta=np.linspace(.01,-.01,10)
        components=np.tile(np.stack([baseline+delta/2,-baseline+delta/2]),(20,1))
        block=split_stability(components,np.repeat(np.arange(20),2),n_splits=100,seed=8)
        item=split_stability(components,np.arange(40),n_splits=100,seed=8)
        self.assertAlmostEqual(block['mean_r'],1.)
        self.assertLess(item['mean_r'],0.)
        self.assertIsNone(item['sb_positive_only'])

    def test_flat_models_are_undefined_not_reliable(self):
        result=split_stability(np.zeros((8,5)),np.arange(8),n_splits=20)
        self.assertEqual(result['valid_splits'],0)
        self.assertIsNone(result['mean_r'])

    def test_weighted_split_matches_expanded_repetitions(self):
        values=np.array([[1.,2.,3.],[2.,1.,4.],[4.,3.,2.],[3.,4.,1.]])
        weights=np.repeat(np.array([1,3,2,1])[:,None],3,axis=1)
        expanded=np.repeat(values,[1,3,2,1],axis=0)
        blocks=np.repeat(np.arange(4),[1,3,2,1])
        weighted=split_stability(values,np.arange(4),weights=weights,n_splits=50,seed=4)
        raw=split_stability(expanded,blocks,n_splits=50,seed=4)
        self.assertEqual(weighted,raw)

    def test_ties_share_best_rank_probability(self):
        mat=pd.DataFrame({'a':[0.,0.,0.],'b':[0.,0.,0.]})
        out,_=bootstrap_models(mat,n_boot=20)
        self.assertEqual(out[0]['bottom_rank_share'],.5)
        self.assertEqual(out[1]['bottom_rank_share'],.5)
        self.assertEqual(out[0]['rank_ci_low'],1.5)

    def test_seed_independent_of_other_analysis_calls(self):
        frame=pd.DataFrame({'a':[1.,2.,3.,4.],'b':[2.,1.,4.,3.]})
        one=concordance(frame,n_perm=99,seed=2,label='x')
        concordance(frame,n_perm=199,seed=3,label='different')
        self.assertEqual(one,concordance(frame,n_perm=99,seed=2,label='x'))


if __name__=='__main__':
    unittest.main()
