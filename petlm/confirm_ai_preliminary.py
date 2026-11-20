"""Record a human confirmation of an automated preliminary score.

This intentionally does not produce a final independent human-score status.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .run import checksum


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preliminary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reviewer-id',required=True)
    parser.add_argument('--signature',required=True)
    parser.add_argument('--reviewed-at',required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    source=read_jsonl(args.preliminary)
    if len(source)!=120 or any(row.get('scoring_method')!='automated_preliminary' for row in source):
        raise RuntimeError('Expected exactly 120 automated preliminary rows.')
    rows=[]
    for row in source:
        rows.append(dict(row,
            reviewer_id=args.reviewer_id,reviewed_at=args.reviewed_at,
            reviewer_signature=args.signature,
            independent_of_model_development=False,
            human_reviewed=True,
            scoring_method='human_confirmed_ai_assisted',
            notes=row['notes']+' 人工确认：同意该自动预评分结果。'))
    args.output.mkdir(parents=True)
    score_path=args.output/'scores.human-confirmed-ai-assisted.jsonl'
    write_jsonl(score_path,rows)
    passed=sum(row['decision']=='pass' for row in rows)
    report={'kind':'petlm_frozen_human_confirmed_ai_assisted_score_v1',
            'status':'human_confirmed_ai_assisted_not_independent_human_score',
            'reviewer_id':args.reviewer_id,'reviewed_at':args.reviewed_at,
            'reviewer_signature':args.signature,'count':len(rows),'passed':passed,
            'failed':len(rows)-passed,'pass_rate':passed/len(rows),
            'independent_of_model_development':False,
            'source_preliminary':str(args.preliminary),
            'source_preliminary_sha256':checksum(args.preliminary),
            'scores_sha256':checksum(score_path),
            'warning':'This report records confirmation of AI-assisted preliminary scores; '
                      'it is not an independent blinded human evaluation.'}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
