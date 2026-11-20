import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from petlm.focused_behavior_v4 import build_blind, build_development
from petlm.routing_regression import evaluate_answer


class FocusedBehaviorV4Tests(unittest.TestCase):
    def test_development_rows_are_isolated_and_not_sft_eligible(self):
        with TemporaryDirectory() as temp:
            output=Path(temp)/'development'
            report=build_development(output,8,123,[])
            self.assertEqual(report['counts'],{'development':16,'internal_test':16})
            self.assertFalse(report['eligible_for_sft'])
            rows=[json.loads(line) for line in (output/'development.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(not row['eligible_for_sft'] for row in rows))
            self.assertTrue(all(evaluate_answer(row,row['proposed_answer'])['passed'] for row in rows))

    def test_blind_is_frozen_pending_review_and_disjoint(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            development=root/'development'
            build_development(development,8,123,[])
            blind=root/'blind'
            report=build_blind(blind,development,80,456,[])
            self.assertEqual(report['status'],'pending_independent_item_review')
            self.assertFalse(report['usable_as_frozen_answer_key'])
            self.assertEqual(report['count'],80)
            self.assertTrue(report['leakage_report']['passed'])

    def test_structured_answer_checks_question_count_and_forbidden_terms(self):
        row={'requirements':[['演讲排练'],['？','?']],
             'forbidden':['建议'],'question_mark_count':1}
        self.assertTrue(evaluate_answer(row,'关于演讲排练，你卡在哪里？')['passed'])
        self.assertFalse(evaluate_answer(row,'关于演讲排练，你卡在哪里？还需要什么？')['passed'])
        self.assertFalse(evaluate_answer(row,'建议先说说演讲排练哪里卡住？')['passed'])


if __name__=='__main__':
    unittest.main()
