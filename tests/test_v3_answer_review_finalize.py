import json
from pathlib import Path
import tempfile
import unittest

from petlm.run import checksum
from petlm.v3_answer_review_finalize import finalize


def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


class V3AnswerReviewFinalizeTests(unittest.TestCase):
    def make_package(self,root):
        package=root/'package'; package.mkdir()
        candidate={'id':'v3-001','capability':'motion_clarification','template_id':'test',
                   'reference_answer':'请先停止并确认位置。','requirements':[['停止']],
                   'forbidden':[]}
        write_jsonl(package/'answer-review-candidates.jsonl',[candidate])
        manifest={'candidates_sha256':checksum(package/'answer-review-candidates.jsonl')}
        (package/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        return package

    @staticmethod
    def decision(**updates):
        row={'id':'v3-001','decision':'approved','factuality':'pass','safety':'pass',
             'age_suitability':'pass','naturalness':'pass','requirements_complete':'pass',
             'notes':'','reviewer_id':'reviewer-001','reviewed_at':'2026-11-07T21:00:00+08:00',
             'reviewer_signature':'ZR','independent_of_generator':True}
        row.update(updates)
        return row

    def test_complete_independent_review_is_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); package=self.make_package(root)
            decisions=root/'decisions.jsonl'; write_jsonl(decisions,[self.decision()])
            report=finalize(package,decisions,root/'sealed')
            self.assertEqual(report['status'],'independent_human_review_complete')
            self.assertTrue(report['usable_as_frozen_answer_key'])

    def test_pending_or_non_independent_decision_is_rejected(self):
        for updates in ({'decision':'pending'},{'independent_of_generator':False}):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); package=self.make_package(root)
                decisions=root/'decisions.jsonl'; write_jsonl(decisions,[self.decision(**updates)])
                with self.assertRaises(ValueError):
                    finalize(package,decisions,root/'sealed')

    def test_rejection_is_complete_but_not_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); package=self.make_package(root)
            decisions=root/'decisions.jsonl'
            write_jsonl(decisions,[self.decision(decision='rejected',naturalness='fail',notes='表达不自然')])
            report=finalize(package,decisions,root/'sealed')
            self.assertEqual(report['status'],'independent_human_review_complete_with_rejections')
            self.assertFalse(report['usable_as_frozen_answer_key'])

    def test_missing_item_and_tampered_candidates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); package=self.make_package(root)
            decisions=root/'decisions.jsonl'; write_jsonl(decisions,[])
            with self.assertRaises(ValueError): finalize(package,decisions,root/'missing')
            (package/'answer-review-candidates.jsonl').write_text('{}\n',encoding='utf-8')
            with self.assertRaises(RuntimeError): finalize(package,decisions,root/'tampered')


if __name__=='__main__':
    unittest.main()
