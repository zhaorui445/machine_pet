"""Split pending human-review datasets into immutable reviewer batches."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from .human_review_pipeline import SPLITS, read_jsonl, write_jsonl
from .run import checksum


def create_assignment(candidates,output,reviewer_id,batch_size=60):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    if not reviewer_id.strip() or batch_size<=0:
        raise ValueError('A reviewer id and positive batch size are required.')
    manifest_path=candidates/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('review_status')!='pending_independent_item_review':
        raise RuntimeError('Candidate package is not pending independent review.')
    splits=tuple(manifest.get('splits',SPLITS))
    if not splits:
        raise RuntimeError('Candidate package must declare at least one split.')
    rows=[]
    for split in splits:
        path=candidates/f'{split}-candidates.jsonl'
        if checksum(path)!=manifest['candidate_files'][split]:
            raise RuntimeError(f'{split} candidate hash changed.')
        rows.extend(read_jsonl(path))
    template_path=candidates/'review-decisions.template.jsonl'
    if checksum(template_path)!=manifest['review_template_sha256']:
        raise RuntimeError('Review template hash changed.')
    templates=read_jsonl(template_path)
    by_id={row['id']:row for row in templates}
    if len(by_id)!=len(templates) or set(by_id)!={row['id'] for row in rows}:
        raise RuntimeError('Candidates and review template do not cover identical unique ids.')
    output.mkdir(parents=True)
    batches=[]
    for start in range(0,len(rows),batch_size):
        number=start//batch_size+1
        batch=output/f'batch-{number:02d}'
        batch.mkdir()
        selected=rows[start:start+batch_size]
        decisions=[]
        for row in selected:
            decision=dict(by_id[row['id']])
            decision['reviewer_id']=reviewer_id
            decisions.append(decision)
        candidate_path=batch/'candidates.jsonl'
        decision_path=batch/'decisions.pending.jsonl'
        write_jsonl(candidate_path,selected)
        write_jsonl(decision_path,decisions)
        batches.append({
            'batch':number,'count':len(selected),
            'first_id':selected[0]['id'],'last_id':selected[-1]['id'],
            'candidates':str(candidate_path),'candidates_sha256':checksum(candidate_path),
            'decisions':str(decision_path),'decisions_sha256':checksum(decision_path),
            'status':'assigned_pending_human_review',
        })
    report={
        'kind':'human_review_assignment_v1','dataset_id':manifest.get('dataset_id'),
        'created_at':datetime.now(timezone.utc).isoformat(),
        'status':'assigned_pending_human_review','reviewer_id':reviewer_id,
        'count':len(rows),'batch_size':batch_size,'batch_count':len(batches),
        'source_candidates':str(candidates),'source_manifest_sha256':checksum(manifest_path),
        'batches':batches,
        'warning':('No decision, rating, time, signature, or independence statement was pre-filled. '
                   'The reviewer must read every item.'),
        'splits': list(splits),
        'isolation':('Development rows may enter SFT only after independent approval. Internal-test '
                     'and holdout rows must remain evaluation-only. No physical actuation is permitted.'),
    }
    (output/'assignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    guide=(
        f'# {manifest.get("dataset_id","dataset")} 人工审核任务\n\n'
        f'共 {len(rows)} 条，分为 {len(batches)} 批，每批最多 {batch_size} 条。\n\n'
        '逐条对照 `candidates.jsonl` 与 `decisions.pending.jsonl`。填写 approved/rejected、'
        '事实性、年龄适配、安全性、自然度、说明、实际审核时间、签名和独立性声明。\n\n'
        '不能批量默认通过。审核完成前，任何候选都不得用于训练；内部测试和隔离保留集永不进入训练。\n'
    )
    (output/'REVIEW_ASSIGNMENT.md').write_text(guide,encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reviewer-id',required=True)
    parser.add_argument('--batch-size',type=int,default=60)
    args=parser.parse_args()
    print(json.dumps(create_assignment(args.candidates,args.output,args.reviewer_id,args.batch_size),
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
