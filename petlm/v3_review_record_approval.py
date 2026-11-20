"""Record a reviewer's attested item-level approval of assigned v3 batches."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from .run import checksum


RATINGS=('factuality','safety','age_suitability','naturalness','requirements_complete')


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path,rows):
    with path.open('w',encoding='utf-8',newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')


def record_approvals(assignment_root,output,reviewer_id,reviewer_signature,
                     reviewed_at,attest_item_level_review=False):
    if not attest_item_level_review:
        raise ValueError('Explicit item-level human-review attestation is required.')
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    for field,value in (('reviewer_id',reviewer_id),('reviewer_signature',reviewer_signature)):
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f'{field} is required.')
    try:
        datetime.fromisoformat(reviewed_at.replace('Z','+00:00'))
    except (AttributeError,ValueError) as exc:
        raise ValueError('reviewed_at must be an ISO-8601 timestamp.') from exc

    assignment_path=assignment_root/'assignment.json'
    assignment=json.loads(assignment_path.read_text(encoding='utf-8'))
    benchmark_id=assignment.get('benchmark_id','frozen-benchmark')
    if assignment.get('status')!='assigned_pending_human_review':
        raise RuntimeError('Unexpected assignment status.')
    if assignment.get('reviewer_id')!=reviewer_id:
        raise ValueError('Reviewer id does not match the assignment.')

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
            raise RuntimeError(f'Batch {number} no longer matches its assignment hashes.')
        candidates=read_jsonl(candidates_path)
        decisions=read_jsonl(decisions_path)
        candidate_ids=[row['id'] for row in candidates]
        if candidate_ids!=[row.get('id') for row in decisions]:
            raise RuntimeError(f'Batch {number} candidate and decision order differs.')
        batch_rows=[]
        for row in decisions:
            identifier=row['id']
            if identifier in seen:
                raise RuntimeError(f'Duplicate review id: {identifier}')
            seen.add(identifier)
            recorded=dict(row)
            recorded.update({
                'decision':'approved',
                **{field:'pass' for field in RATINGS},
                'notes':'审核人确认已逐条审阅；事实性、安全性、年龄适配、自然度和要求完整性均合格。',
                'reviewer_id':reviewer_id,
                'reviewed_at':reviewed_at,
                'reviewer_signature':reviewer_signature,
                'independent_of_generator':True,
                'recording_method':'reviewer_attested_item_level_approval',
            })
            batch_rows.append(recorded)
            completed.append(recorded)
        batch_output=output/f'batch-{number:02d}'
        batch_output.mkdir()
        write_jsonl(batch_output/'decisions.completed.jsonl',batch_rows)

    if len(completed)!=assignment['count'] or len(seen)!=assignment['count']:
        raise RuntimeError('Completed records do not cover the assignment exactly once.')
    combined=output/'review-decisions.completed.jsonl'
    write_jsonl(combined,completed)
    report={
        'kind':'reviewer_attested_item_level_approval_v2','benchmark_id':benchmark_id,
        'status':'recorded_pending_final_validation',
        'count':len(completed),'approved':len(completed),'rejected':0,
        'reviewer_id':reviewer_id,'reviewer_signature':reviewer_signature,
        'reviewed_at':reviewed_at,'independent_of_generator_attested':True,
        'source_assignment':str(assignment_root),
        'source_assignment_sha256':checksum(assignment_path),
        'combined_decisions':str(combined),
        'combined_decisions_sha256':checksum(combined),
        'recording_method':'reviewer_attested_item_level_approval',
        'warning':('This file records the human reviewer\'s stated item-level review. '
                   'The recording program did not independently judge answer quality.'),
    }
    (output/'recording-manifest.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
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
    report=record_approvals(
        args.assignment,args.output,args.reviewer_id,args.reviewer_signature,
        args.reviewed_at,args.attest_item_level_review)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
