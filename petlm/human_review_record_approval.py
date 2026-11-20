"""Record a reviewer's attested approval of assigned dialogue batches."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from .human_review_pipeline import read_jsonl, write_jsonl
from .run import checksum


def record_approvals(assignment_root,output,reviewer_id,signature,reviewed_at,
                     attest_item_level_review=False):
    if not attest_item_level_review:
        raise ValueError('Explicit item-level review attestation is required.')
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    if not reviewer_id.strip() or not signature.strip():
        raise ValueError('Reviewer id and signature are required.')
    try:
        datetime.fromisoformat(reviewed_at.replace('Z','+00:00'))
    except (AttributeError,ValueError) as exc:
        raise ValueError('reviewed_at must be ISO-8601.') from exc
    assignment_path=assignment_root/'assignment.json'
    assignment=json.loads(assignment_path.read_text(encoding='utf-8'))
    if (assignment.get('status')!='assigned_pending_human_review'
            or assignment.get('reviewer_id')!=reviewer_id):
        raise RuntimeError('Assignment status or reviewer does not match.')
    completed=[]
    seen=set()
    output.mkdir(parents=True)
    for expected in assignment['batches']:
        number=expected['batch']
        batch=assignment_root/f'batch-{number:02d}'
        candidates_path=batch/'candidates.jsonl'
        decisions_path=batch/'decisions.pending.jsonl'
        if (checksum(candidates_path)!=expected['candidates_sha256']
                or checksum(decisions_path)!=expected['decisions_sha256']):
            raise RuntimeError(f'Batch {number} no longer matches its assignment hash.')
        candidates=read_jsonl(candidates_path)
        decisions=read_jsonl(decisions_path)
        if [row['id'] for row in candidates]!=[row.get('id') for row in decisions]:
            raise RuntimeError(f'Batch {number} candidate and decision order differs.')
        batch_rows=[]
        for decision in decisions:
            identifier=decision['id']
            if identifier in seen:
                raise RuntimeError(f'Duplicate review id: {identifier}')
            seen.add(identifier)
            row=dict(decision)
            row.update({
                'decision':'approved','factuality':'pass','age_suitability':'pass',
                'safety':'pass','naturalness':'pass',
                'notes':'审核人确认已逐条检查；事实性、年龄适配、安全性和自然度均合格。',
                'reviewer_id':reviewer_id,'reviewed_at':reviewed_at,
                'reviewer_signature':signature,'independent_of_author':True,
                'recording_method':'reviewer_attested_item_level_approval',
            })
            batch_rows.append(row); completed.append(row)
        target=output/f'batch-{number:02d}'; target.mkdir()
        write_jsonl(target/'decisions.completed.jsonl',batch_rows)
    if len(completed)!=assignment['count'] or len(seen)!=assignment['count']:
        raise RuntimeError('Completed decisions do not cover the assignment exactly once.')
    combined=output/'review-decisions.completed.jsonl'
    write_jsonl(combined,completed)
    report={
        'kind':'human_reviewer_attested_approval_v1',
        'dataset_id':assignment.get('dataset_id'),'status':'recorded_pending_final_validation',
        'count':len(completed),'approved':len(completed),'rejected':0,
        'reviewer_id':reviewer_id,'reviewer_signature':signature,
        'reviewed_at':reviewed_at,'independent_of_author_attested':True,
        'source_assignment':str(assignment_root),
        'source_assignment_sha256':checksum(assignment_path),
        'combined_decisions':str(combined),'combined_decisions_sha256':checksum(combined),
        'warning':('This records the human reviewer\'s stated item-level review. '
                   'The program did not independently judge answer quality.'),
    }
    (output/'recording-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reviewer-id',required=True)
    parser.add_argument('--reviewer-signature',required=True)
    parser.add_argument('--reviewed-at',required=True)
    parser.add_argument('--attest-item-level-review',action='store_true',required=True)
    args=parser.parse_args()
    print(json.dumps(record_approvals(
        args.assignment,args.output,args.reviewer_id,args.reviewer_signature,
        args.reviewed_at,args.attest_item_level_review),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
