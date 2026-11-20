"""Freeze a reviewed behavior dataset after explicit user approval."""
import argparse
from datetime import date
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    args.output.mkdir(parents=True)
    reviewed_at=str(date.today())
    counts={}
    for split in ('train','validation','test'):
        rows=[]
        with (args.input/f'{split}.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row=json.loads(line)
                row['human_review_status']='approved_by_user'
                row['user_approval']='本轮对话中用户确认审核通过'
                row['user_approved_at']=reviewed_at
                row['review_scope']='workspace_item_review_plus_user_approval'
                row['training_group']='human_approved_behavior_v1'
                rows.append(row)
        counts[split]=len(rows)
        with (args.output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    shutil.copy2(args.input/'review-decisions.jsonl',args.output/'review-decisions.codex.jsonl')
    source=json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
    manifest=dict(source,reviewed_at=reviewed_at,
                  human_review_status='approved_by_user',
                  user_approval='本轮对话中用户确认审核通过',
                  review_scope='workspace_item_review_plus_user_approval',
                  counts=counts,
                  warning='用户已确认审核通过；逐条复核由工作区检查完成，未声称外部独立人工审计。')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
