import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from petlm.human_review_assignment import create_assignment
from petlm.human_review_pipeline import REVIEW_SCHEMA, write_jsonl
from petlm.run import checksum


class HumanReviewAssignmentTests(unittest.TestCase):
    def test_assignment_preserves_pending_human_fields(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.mkdir()
            rows=[]; templates=[]
            for split in ('development','internal_test','frozen'):
                row={'id':f'v5-{split}','split':split,'prompt':'p','proposed_answer':'a'}
                rows.append(row); write_jsonl(source/f'{split}-candidates.jsonl',[row])
                templates.append({'schema':REVIEW_SCHEMA,'id':row['id'],'decision':'pending',
                                  'reviewer_id':'','reviewed_at':'','independent_of_author':False,
                                  'factuality':'','age_suitability':'','safety':'','naturalness':'',
                                  'approve_for_split':split,'notes':'','reviewer_signature':''})
            write_jsonl(source/'review-decisions.template.jsonl',templates)
            manifest={'dataset_id':'v5','review_status':'pending_independent_item_review',
                      'candidate_files':{s:checksum(source/f'{s}-candidates.jsonl')
                                         for s in ('development','internal_test','frozen')},
                      'review_template_sha256':checksum(source/'review-decisions.template.jsonl')}
            (source/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            report=create_assignment(source,root/'assigned','reviewer-001',2)
            self.assertEqual(report['batch_count'],2)
            decisions=read_jsonl(root/'assigned'/'batch-01'/'decisions.pending.jsonl')
            self.assertTrue(all(row['reviewer_id']=='reviewer-001' for row in decisions))
            self.assertTrue(all(row['decision']=='pending' and not row['reviewed_at'] for row in decisions))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


if __name__=='__main__':
    unittest.main()
