import json
from pathlib import Path
import tempfile
import unittest

from petlm.run import checksum
from petlm.v3_review_assignment import create_assignment


def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


class V3ReviewAssignmentTests(unittest.TestCase):
    def test_assignment_splits_batches_without_prefilling_human_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); package=root/'package'; package.mkdir()
            candidates=[]; templates=[]
            for index in range(3):
                identifier=f'v3-{index}'
                candidates.append({'id':identifier,'reference_answer':'回答'})
                templates.append({'id':identifier,'decision':'pending','factuality':'','safety':'',
                                  'age_suitability':'','naturalness':'','requirements_complete':'',
                                  'notes':'','reviewer_id':'','reviewed_at':'',
                                  'reviewer_signature':'','independent_of_generator':False})
            write_jsonl(package/'answer-review-candidates.jsonl',candidates)
            write_jsonl(package/'review-decisions.template.jsonl',templates)
            manifest={'candidates_sha256':checksum(package/'answer-review-candidates.jsonl'),
                      'review_template_sha256':checksum(package/'review-decisions.template.jsonl')}
            (package/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            report=create_assignment(package,root/'assigned','reviewer-001',2)
            self.assertEqual(report['batch_count'],2)
            rows=[json.loads(line) for line in
                  (root/'assigned'/'batch-01'/'decisions.pending.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(row['reviewer_id']=='reviewer-001' for row in rows))
            self.assertTrue(all(row['decision']=='pending' and not row['reviewed_at']
                                and not row['reviewer_signature']
                                and row['independent_of_generator'] is False for row in rows))

    def test_tampered_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); package=root/'package'; package.mkdir()
            write_jsonl(package/'answer-review-candidates.jsonl',[{'id':'v3-1'}])
            write_jsonl(package/'review-decisions.template.jsonl',[{'id':'v3-1'}])
            (package/'manifest.json').write_text(json.dumps({
                'candidates_sha256':'wrong','review_template_sha256':'wrong'}),encoding='utf-8')
            with self.assertRaises(RuntimeError):
                create_assignment(package,root/'assigned','reviewer-001',50)


if __name__=='__main__':
    unittest.main()
