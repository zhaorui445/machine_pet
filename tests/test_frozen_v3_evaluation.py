import unittest

from petlm.frozen_v3_evaluation import score_mode


class FrozenV3EvaluationTests(unittest.TestCase):
    def test_score_mode_checks_requirements_and_forbidden_terms(self):
        keys=[{'id':'a','template_id':'one','capability':'format_control',
               'requirements':[['结论'],['1.','1、']],'forbidden':['总结']},
              {'id':'b','template_id':'two','capability':'safety_behavior',
               'requirements':[['不能','不可以']],'forbidden':[]}]
        predictions=[{'id':'a','answer':'结论：先做。\n1. 第一步。','response_source':'rule'},
                     {'id':'b','answer':'可以直接尝试。','response_source':'model'}]
        result=score_mode(predictions,keys)
        self.assertEqual(result['passed'],1)
        self.assertEqual(result['total'],2)
        self.assertEqual(len(result['failures']),1)

    def test_score_mode_requires_exact_id_coverage(self):
        keys=[{'id':'a','template_id':'one','capability':'format_control',
               'requirements':[],'forbidden':[]}]
        with self.assertRaises(RuntimeError):
            score_mode([],keys)

    def test_score_mode_enforces_exact_question_count(self):
        keys=[{'id':'one','template_id':'single_question_limit','capability':'format_control',
               'requirements':[['？']],'forbidden':[],'question_mark_count':1}]
        self.assertEqual(score_mode([{'id':'one','answer':'你想先处理哪一点？'}],keys)['passed'],1)
        result=score_mode([{'id':'one','answer':'你想先处理哪一点？什么时候开始？'}],keys)
        self.assertEqual(result['passed'],0)
        self.assertFalse(result['failures'][0]['question_count_ok'])


if __name__=='__main__':
    unittest.main()
