"""Validate and seal independent item-level review decisions for v3 answers."""
import argparse
from collections import Counter
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


def _review_time(value):
    if not isinstance(value,str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:
        return False
    return True


def validate_decision(row):
    required={'id','decision',*RATINGS,'notes','reviewer_id','reviewed_at',
              'reviewer_signature','independent_of_generator'}
    missing=required-set(row)
    if missing:
        raise ValueError(f'{row.get("id","unknown")}: missing fields {sorted(missing)}')
    if row['decision'] not in ('approved','rejected'):
        raise ValueError(f'{row["id"]}: decision must be approved or rejected.')
    for field in RATINGS:
        if row[field] not in ('pass','fail'):
            raise ValueError(f'{row["id"]}: {field} must be pass or fail.')
    if row['decision']=='approved' and any(row[field]!='pass' for field in RATINGS):
        raise ValueError(f'{row["id"]}: an approved row requires every rating to pass.')
    if row['decision']=='rejected' and not any(row[field]=='fail' for field in RATINGS):
        raise ValueError(f'{row["id"]}: a rejected row requires at least one failed rating.')
    if row['decision']=='rejected' and (not isinstance(row['notes'],str) or not row['notes'].strip()):
        raise ValueError(f'{row["id"]}: a rejected row requires notes.')
    for field in ('reviewer_id','reviewer_signature'):
        if not isinstance(row[field],str) or not row[field].strip():
            raise ValueError(f'{row["id"]}: {field} is required.')
    if not _review_time(row['reviewed_at']):
        raise ValueError(f'{row["id"]}: reviewed_at must be an ISO-8601 timestamp.')
    if row['independent_of_generator'] is not True:
        raise ValueError(f'{row["id"]}: reviewer independence must be explicitly true.')


def finalize(package,decisions_path,output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    candidates_path=package/'answer-review-candidates.jsonl'
    if checksum(candidates_path)!=manifest['candidates_sha256']:
        raise RuntimeError('Candidate review file no longer matches its sealed manifest.')
    candidates=read_jsonl(candidates_path)
    source_keys={}
    source_key_path=manifest.get('source_answer_key')
    if source_key_path:
        source_key_path=Path(source_key_path)
        if checksum(source_key_path)!=manifest.get('source_answer_key_sha256'):
            raise RuntimeError('Source answer key no longer matches its sealed review-package hash.')
        source_keys={row['id']:row for row in read_jsonl(source_key_path)}
    decisions=read_jsonl(decisions_path)
    candidate_ids=[row['id'] for row in candidates]
    decision_ids=[row.get('id') for row in decisions]
    if len(candidate_ids)!=len(set(candidate_ids)) or len(decision_ids)!=len(set(decision_ids)):
        raise ValueError('Candidate and decision ids must be unique.')
    if set(candidate_ids)!=set(decision_ids):
        missing=sorted(set(candidate_ids)-set(decision_ids))
        extra=sorted(set(decision_ids)-set(candidate_ids))
        raise ValueError(f'Decisions must cover every item exactly once: missing={missing[:5]}, extra={extra[:5]}')
    by_id={row['id']:row for row in decisions}
    for identifier in candidate_ids:
        validate_decision(by_id[identifier])
    ordered=[by_id[identifier] for identifier in candidate_ids]
    counts=Counter(row['decision'] for row in ordered)
    reviewer_ids=sorted({row['reviewer_id'] for row in ordered})
    reviewed=[]
    for candidate in candidates:
        decision=by_id[candidate['id']]
        if decision['decision']=='approved':
            item={'id':candidate['id'],'capability':candidate['capability'],
                  'template_id':candidate['template_id'],
                  'answer':candidate['reference_answer'],
                  'requirements':candidate['requirements'],
                  'forbidden':candidate['forbidden'],
                  'reviewer_id':decision['reviewer_id'],
                  'reviewed_at':decision['reviewed_at'],
                  'reviewer_signature':decision['reviewer_signature']}
            source=source_keys.get(candidate['id'],{})
            if source.get('question_mark_count') is not None:
                item['question_mark_count']=source['question_mark_count']
            reviewed.append(item)
    output.mkdir(parents=True)
    write_jsonl(output/'review-decisions.jsonl',ordered)
    write_jsonl(output/'answer-key.reviewed.jsonl',reviewed)
    all_approved=counts['approved']==len(candidates)
    report={'kind':'independent_answer_review_v2',
            'benchmark_id':manifest.get('benchmark_id','frozen-benchmark'),
            'status':('independent_human_review_complete' if all_approved
                      else 'independent_human_review_complete_with_rejections'),
            'count':len(candidates),'approved':counts['approved'],'rejected':counts['rejected'],
            'reviewer_ids':reviewer_ids,'item_level_review':True,
            'independence_attested':True,'source_package':str(package),
            'source_manifest_sha256':checksum(package/'manifest.json'),
            'decisions_input_sha256':checksum(decisions_path),
            'sealed_decisions_sha256':checksum(output/'review-decisions.jsonl'),
            'reviewed_answer_key_sha256':checksum(output/'answer-key.reviewed.jsonl'),
            'usable_as_frozen_answer_key':all_approved,
            'warning':('This validator checks record completeness and consistency; it does not '
                       'substitute for the human reviewer reading each item.')}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--decisions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(finalize(args.package,args.decisions,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
