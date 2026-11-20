import json
from pathlib import Path
import tempfile
import unittest

from petlm.run import checksum
from petlm.v3_review_record_approval import record_approvals


def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


class V3ReviewRecordApprovalTests(unittest.TestCase):
    def make_assignment(self,root):
        assigned=root/'assigned'; batch=assigned/'batch-01'; batch.mkdir(parents=True)
        candidates=[{'id':'v3-1'},{'id':'v3-2'}]
        template=lambda identifier: {
            'id':identifier,'decision':'pending','factuality':'','safety':'',
            'age_suitability':'','naturalness':'','requirements_complete':'','notes':'',
            'reviewer_id':'reviewer-001','reviewed_at':'','reviewer_signature':'',
            'independent_of_generator':False}
        decisions=[template('v3-1'),template('v3-2')]
        write_jsonl(batch/'candidates.jsonl',candidates)
        write_jsonl(batch/'decisions.pending.jsonl',decisions)
        assignment={
            'status':'assigned_pending_human_review','reviewer_id':'reviewer-001',
            'count':2,'batches':[{'batch':1,
                'candidates_sha256':checksum(batch/'candidates.jsonl'),
                'decisions_sha256':checksum(batch/'decisions.pending.jsonl')}],}
        (assigned/'assignment.json').write_text(json.dumps(assignment),encoding='utf-8')
        return assigned

    def test_records_attested_approvals_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); assigned=self.make_assignment(root)
            source_hash=checksum(assigned/'batch-01'/'decisions.pending.jsonl')
            report=record_approvals(assigned,root/'completed','reviewer-001','ZR',
                                    '2026-11-07T21:00:00+08:00',True)
            rows=read_jsonl(root/'completed'/'review-decisions.completed.jsonl')
            self.assertEqual(report['approved'],2)
            self.assertTrue(all(row['decision']=='approved' for row in rows))
            self.assertTrue(all(row['factuality']=='pass' for row in rows))
            self.assertEqual(checksum(assigned/'batch-01'/'decisions.pending.jsonl'),source_hash)

    def test_requires_attestation_and_matching_reviewer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); assigned=self.make_assignment(root)
            with self.assertRaises(ValueError):
                record_approvals(assigned,root/'no-attest','reviewer-001','ZR',
                                 '2026-11-07T21:00:00+08:00',False)
            with self.assertRaises(ValueError):
                record_approvals(assigned,root/'wrong-reviewer','reviewer-002','ZR',
                                 '2026-11-07T21:00:00+08:00',True)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


if __name__=='__main__':
    unittest.main()
