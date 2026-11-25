import sys
import unittest
from pathlib import Path
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
from ber.metrics import macro_f05, bootstrap_ci
from ber.normalize import build_views
from synthetic_matcher import macro_f05_reference, run


class ResearchTests(unittest.TestCase):
    def test_metric_with_singleton_and_false_positive(self):
        q=pl.DataFrame({'q':[1,2,3]})
        truth=pl.DataFrame({'q':[1,1,2],'d':[10,11,20]})
        pred=pl.DataFrame({'q':[1,2,2,2],'d':[10,20,99,99]})
        expected=(1.25/1.5 + 1.25/2.25 + 1)/3
        self.assertAlmostEqual(macro_f05(pred,truth,q),expected)

    def test_empty_prediction(self):
        q=pl.DataFrame({'q':[1,2]})
        truth=pl.DataFrame({'q':[1],'d':[10]})
        empty=truth.head(0)
        self.assertAlmostEqual(macro_f05(empty,truth,q),0.5)
        self.assertAlmostEqual(macro_f05(empty,empty,q),1.0)

    def test_bootstrap_constant(self):
        self.assertEqual(bootstrap_ci(np.ones(8),n=50,seed=7),(1.0,1.0))

    def test_normalization_aliases(self):
        frame=pl.DataFrame({'entity_id':['S1_1','S1_2'], 'business_name':['Cedar Pvt Ltd','CEDAR Private Limited'],
                           'business_address':['012 Oak Rd','12 Oak Road'], 'country':['IN','IN']})
        views=build_views(frame)
        self.assertEqual(views['name_tok'].to_list()[0],views['name_tok'].to_list()[1])
        self.assertEqual(views['addr_num'].to_list(),[['12'],['12']])
        self.assertEqual(views['business_name'].to_list(),frame['business_name'].to_list())

    def test_reference_metric_matches_production(self):
        y=np.array([1,0,1,0,0,0]); pred=np.array([1,1,0,0,0,0]); groups=np.array([1,1,2,2,3,3]); ids=np.arange(len(y))
        q=pl.DataFrame({'q':np.unique(groups)})
        truth=pl.DataFrame({'q':groups[y==1],'d':ids[y==1]})
        p=pl.DataFrame({'q':groups[pred==1],'d':ids[pred==1]})
        self.assertAlmostEqual(macro_f05(p,truth,q),macro_f05_reference(y,pred,groups))

    def test_synthetic_training(self):
        r=run()
        self.assertEqual(r['test_groups'],80)
        self.assertEqual(r['scope'],'SYNTHETIC_CPU_DEMO_NOT_COMPETITION')
        for key in ('validation_macro_f05','test_baseline_macro_f05','test_lightgbm_macro_f05'):
            self.assertGreaterEqual(r[key],0.0)
            self.assertLessEqual(r[key],1.0)
        self.assertGreater(r['test_lightgbm_macro_f05'],0.5)

if __name__=='__main__': unittest.main()
