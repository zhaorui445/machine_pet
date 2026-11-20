import json
from pathlib import Path
import tempfile
import unittest

from petlm.human_review_pipeline import (AUTHOR_ID, build_candidates, finalize,
                                          leakage_report, normalized)


class HumanReviewPipelineTests(unittest.TestCase):
    def test_candidates_are_pending_and_three_way_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=root/'candidates'
            build_candidates(output,count_per_family=9,seed=20261110)
            self.assertTrue((output/'review-decisions.template.jsonl').exists())
            self.assertTrue((output/'frozen-prompts.pending.jsonl').exists())
            self.assertTrue((output/'frozen-answer-key.pending.jsonl').exists())
            rows=[]
            for split in ('development','internal_test','frozen'):
                rows.extend(json.loads(line) for line in
                            (output/f'{split}-candidates.jsonl').read_text(encoding='utf8').splitlines())
            self.assertEqual(len(rows),72)
            self.assertTrue(all(row['review_status']=='pending' for row in rows))
            self.assertTrue(all(row['author_id']==AUTHOR_ID for row in rows))
            public=[json.loads(line) for line in
                    (output/'frozen-prompts.pending.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertTrue(all('proposed_answer' not in row and 'review_checks' not in row
                                and 'prompt' not in row for row in public))

    def test_leakage_detects_exact_and_near_duplicate(self):
        rows=[{'id':'a','prompt':'用户：请停止移动并确认传感器。'},
              {'id':'b','prompt':'用户：请停止移动并确认传感器。'}]
        report=leakage_report(rows)
        self.assertFalse(report['passed'])
        self.assertTrue(any(item['kind']=='exact_internal' for item in report['findings']))
        external=Path(tempfile.mktemp(suffix='.jsonl'))
        try:
            external.write_text(json.dumps({'id':'e','prompt':'用户：请停止移动并确认传感器状态。'},ensure_ascii=False)+'\n',encoding='utf8')
            report=leakage_report([rows[0]], [external], threshold=0.8)
            self.assertFalse(report['passed'])
            self.assertTrue(any(item['kind']=='near_external' for item in report['findings']))
        finally:
            external.unlink(missing_ok=True)

    def test_finalize_requires_independent_complete_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            candidates=root/'candidates'
            build_candidates(candidates,count_per_family=9,seed=20261110)
            template=[json.loads(line) for line in
                      (candidates/'review-decisions.template.jsonl').read_text(encoding='utf8').splitlines()]
            decisions=root/'decisions.jsonl'
            for row in template:
                row.update(decision='approved',reviewer_id='human-001',reviewed_at='2026-11-10',
                           independent_of_author=True,factuality='pass',age_suitability='pass',
                           safety='pass',naturalness='pass',notes='逐条核验通过',
                           reviewer_signature='human-001:example-signature')
            decisions.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in template)+'\n',encoding='utf8')
            output=root/'final'
            finalize(candidates,decisions,output,'human-001')
            self.assertTrue((output/'frozen/prompts.jsonl').exists())
            self.assertTrue((output/'frozen/answer-key.reviewed.jsonl').exists())

    def test_finalize_preserves_extended_frozen_scoring_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            candidates=root/'candidates'
            build_candidates(candidates,count_per_family=9,seed=20261110)
            frozen_path=candidates/'frozen-candidates.jsonl'
            rows=[json.loads(line) for line in frozen_path.read_text(encoding='utf8').splitlines()]
            rows[0].update(requirements=[['必须项']],forbidden=['禁止项'],question_mark_count=1)
            frozen_path.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n',encoding='utf8')
            manifest=json.loads((candidates/'manifest.json').read_text(encoding='utf8'))
            from petlm.run import checksum
            manifest['candidate_files']['frozen']=checksum(frozen_path)
            (candidates/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf8')
            template=[json.loads(line) for line in
                      (candidates/'review-decisions.template.jsonl').read_text(encoding='utf8').splitlines()]
            for row in template:
                row.update(decision='approved',reviewer_id='human-001',reviewed_at='2026-11-10',
                           independent_of_author=True,factuality='pass',age_suitability='pass',
                           safety='pass',naturalness='pass',notes='逐条核验通过',
                           reviewer_signature='signature')
            decisions=root/'decisions.jsonl'
            decisions.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in template)+'\n',encoding='utf8')
            output=root/'final'
            finalize(candidates,decisions,output,'human-001')
            keys=[json.loads(line) for line in
                  (output/'frozen/answer-key.reviewed.jsonl').read_text(encoding='utf8').splitlines()]
            selected=next(row for row in keys if row['id']==rows[0]['id'])
            self.assertEqual(selected['requirements'],[['必须项']])
            self.assertEqual(selected['forbidden'],['禁止项'])
            self.assertEqual(selected['question_mark_count'],1)

    def test_author_cannot_finalize_own_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            candidates=root/'candidates'
            build_candidates(candidates,count_per_family=9,seed=20261110)
            template=[json.loads(line) for line in
                      (candidates/'review-decisions.template.jsonl').read_text(encoding='utf8').splitlines()]
            for row in template:
                row.update(decision='rejected',reviewer_id=AUTHOR_ID,reviewed_at='2026-11-10',
                           factuality='fail',age_suitability='fail',safety='fail',naturalness='fail',
                           notes='same author',reviewer_signature='signature')
            decisions=root/'decisions.jsonl'
            decisions.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in template)+'\n',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'independent'):
                finalize(candidates,decisions,root/'final',AUTHOR_ID)


if __name__=='__main__':
    unittest.main()
