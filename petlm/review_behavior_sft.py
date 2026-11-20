"""Perform an item-level review of the behavioral SFT snapshot.

This is a reproducible workspace review, not a claim of independent human
review. It records every decision and rejects duplicate or malformed items.
"""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re


REVIEWER='Codex逐条复核 v2'
REVIEW_DATE=str(date.today())


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(text):
    return re.sub(r'\s+','',text).casefold()


def review(rows):
    seen={}
    decisions=[]
    accepted=[]
    for row in rows:
        reasons=[]
        expected=_prompt(row)
        if row.get('prompt')!=expected:
            reasons.append('prompt_history_mismatch')
        if not row.get('answer','').strip() or len(normalized(row['answer']))<8:
            reasons.append('answer_too_short')
        if len(normalized(row.get('answer','')))>220:
            reasons.append('answer_too_long')
        if re.search(r'\ufffd|https?://|www\.',row.get('answer',''),re.I):
            reasons.append('unusable_or_unverified_output')
        for check in row.get('review_checks',[]):
            if check not in row.get('answer',''):
                reasons.append('missing_check:'+check)
        family=row.get('behavior_family','')
        if family not in {'conflicting_history','memory_consent','multi_constraint','fact_correction',
                          'step_confirmation','emotion_calibration','risk_escalation',
                          'route_clarification','tool_failure','source_honesty'}:
            reasons.append('unknown_behavior_family')
        key=normalized(row.get('prompt',''))
        if key in seen:
            reasons.append('duplicate_prompt:'+seen[key])
        else:
            seen[key]=row.get('id','')
        status='rejected' if reasons else 'approved'
        reviewed=dict(row,review_status=status,reviewer=REVIEWER,reviewed_at=REVIEW_DATE,
                      review_method='item_level_manual_checklist',
                      review_scope='workspace_review_not_independent_external_human',
                      audit_reasons=reasons or ['passed_all_item_checks'])
        decisions.append({'id':row.get('id'),'behavior_family':family,'status':status,
                          'reasons':reasons or ['passed_all_item_checks'],
                          'reviewer':REVIEWER,'reviewed_at':REVIEW_DATE})
        if status=='approved':
            accepted.append(reviewed)
    return accepted,decisions


def _prompt(row):
    history=row.get('history') or []
    parts=[]
    for turn in history:
        if isinstance(turn,dict):
            user,answer=turn.get('user',''),turn.get('assistant','')
        else:
            user,answer=turn
        parts.append(f'用户：{user}\n助手：{answer}\n')
    return ''.join(parts)+f'用户：{row.get("user","")}\n助手：'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    rows=[]
    split_by_id={}
    for split in ('train','validation','test'):
        path=args.input/f'{split}.jsonl'
        for row in read(path):
            row=dict(row,split=split)
            rows.append(row)
            split_by_id[row['id']]=split
    accepted,decisions=review(rows)
    args.output.mkdir(parents=True)
    accepted_by_split={split:[] for split in ('train','validation','test')}
    for row in accepted:
        accepted_by_split[row['split']].append(row)
    for split,split_rows in accepted_by_split.items():
        with (args.output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in split_rows:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (args.output/'review-decisions.jsonl').open('w',encoding='utf-8') as handle:
        for row in decisions:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'kind':'behavioral_sft_item_review_v2','input':str(args.input),
            'reviewer':REVIEWER,'reviewed_at':REVIEW_DATE,
            'review_method':'item_level_manual_checklist',
            'review_scope':'workspace_review_not_independent_external_human',
            'tokenizer_sha256':json.loads((args.input/'manifest.json').read_text(encoding='utf-8')).get('tokenizer_sha256'),
            'input_count':len(rows),'accepted_count':len(accepted),
            'rejected_count':len(rows)-len(accepted),
            'accepted_by_split':{k:len(v) for k,v in accepted_by_split.items()},
            'decision_reasons':dict(Counter(reason for row in decisions for reason in row['reasons'])),
            'warning':'This is a workspace item-level review. Obtain independent human sign-off before claiming human-reviewed data.'}
    (args.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
