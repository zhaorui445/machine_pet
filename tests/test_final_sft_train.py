import random
import unittest

from petlm.final_sft_train import (GROUP_ORDER, choose_group, group_guard_results,
                                   learning_rate_for_step, sample_batch, validate_protocol,
                                   weighted_score)


class FinalSftTrainTests(unittest.TestCase):
    def setUp(self):
        self.weights={'natural':0.8,'safety':0.05,'memory':0.05,'format':0.05,'motion':0.05}

    def test_deterministic_group_sampling(self):
        first=random.Random(123)
        second=random.Random(123)
        self.assertEqual([choose_group(first,self.weights) for _ in range(1000)],
                         [choose_group(second,self.weights) for _ in range(1000)])

    def test_sample_batch_uses_all_independent_pools(self):
        pools={group:[group] for group in GROUP_ORDER}
        pairs,groups=sample_batch(pools,self.weights,10000,random.Random(99))
        self.assertEqual(pairs,groups)
        counts={group:groups.count(group) for group in GROUP_ORDER}
        self.assertTrue(all(counts[group]>0 for group in GROUP_ORDER))
        self.assertAlmostEqual(counts['natural']/len(groups),0.8,delta=0.02)

    def test_fixed_learning_rate_schedule(self):
        training={'optimizer':{'learning_rate':2e-5},'warmup_steps':100,'maximum_steps':2000,
                  'schedule':{'minimum_learning_rate_ratio':0.2}}
        self.assertAlmostEqual(learning_rate_for_step(training,1),2e-7)
        self.assertAlmostEqual(learning_rate_for_step(training,100),2e-5)
        self.assertAlmostEqual(learning_rate_for_step(training,2000),4e-6)

    def test_weighted_score_and_group_guards(self):
        values={group:1.0 for group in GROUP_ORDER}
        baseline={group:1.0 for group in GROUP_ORDER}
        scoring={'natural':0.6,'safety':0.1,'memory':0.1,'format':0.1,'motion':0.1}
        self.assertAlmostEqual(weighted_score(values,scoring),1.0)
        values['natural']=1.021
        limits={'natural':0.02,'safety':0.05,'memory':0.05,'format':0.05,'motion':0.05}
        guards=group_guard_results(values,baseline,limits)
        self.assertFalse(guards['natural'])
        self.assertTrue(all(guards[group] for group in GROUP_ORDER[1:]))


if __name__=='__main__':
    unittest.main()
