"""Create human-review batches without making or pre-filling review decisions."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from .run import checksum


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path,rows):
    with path.open('w',encoding='utf-8',newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def create_assignment(package,output,reviewer_id,batch_size):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    if not reviewer_id.strip() or batch_size<=0:
        raise ValueError('A reviewer id and positive batch size are required.')
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    benchmark_id=manifest.get('benchmark_id','frozen-benchmark')
    candidates_path=package/'answer-review-candidates.jsonl'
    template_path=package/'review-decisions.template.jsonl'
    if checksum(candidates_path)!=manifest['candidates_sha256']:
        raise RuntimeError('Candidate file hash does not match the package manifest.')
    if checksum(template_path)!=manifest['review_template_sha256']:
        raise RuntimeError('Decision template hash does not match the package manifest.')
    candidates=read_jsonl(candidates_path)
    templates=read_jsonl(template_path)
    template_by_id={row['id']:row for row in templates}
    if len(template_by_id)!=len(templates) or set(template_by_id)!={row['id'] for row in candidates}:
        raise RuntimeError('Candidates and decision templates do not cover identical unique ids.')
    output.mkdir(parents=True)
    batches=[]
    for start in range(0,len(candidates),batch_size):
        number=start//batch_size+1
        batch=output/f'batch-{number:02d}'
        batch.mkdir()
        selected=candidates[start:start+batch_size]
        decisions=[]
        for candidate in selected:
            decision=dict(template_by_id[candidate['id']])
            decision['reviewer_id']=reviewer_id
            # Review time, signature, decisions, ratings, and independence must
            # be supplied by the human reviewer after reading the item.
            decisions.append(decision)
        write_jsonl(batch/'candidates.jsonl',selected)
        write_jsonl(batch/'decisions.pending.jsonl',decisions)
        batches.append({
            'batch':number,'count':len(selected),
            'first_id':selected[0]['id'],'last_id':selected[-1]['id'],
            'candidates':str(batch/'candidates.jsonl'),
            'candidates_sha256':checksum(batch/'candidates.jsonl'),
            'decisions':str(batch/'decisions.pending.jsonl'),
            'decisions_sha256':checksum(batch/'decisions.pending.jsonl'),
            'status':'assigned_pending_human_review',
        })
    report={
        'kind':'independent_human_review_assignment_v2','benchmark_id':benchmark_id,
        'created_at':datetime.now(timezone.utc).isoformat(),
        'status':'assigned_pending_human_review','reviewer_id':reviewer_id,
        'count':len(candidates),'batch_size':batch_size,'batch_count':len(batches),
        'source_package':str(package),'source_manifest_sha256':checksum(package/'manifest.json'),
        'batches':batches,
        'completion_requirements':[
            'Read every candidate and reference answer without consulting model-training staff.',
            'Fill approved/rejected and all five pass/fail ratings item by item.',
            'Fill notes for every rejection.',
            'Enter the actual ISO-8601 review time and reviewer signature.',
            'Set independent_of_generator=true only when the reviewer can personally attest it.',
            'Merge all completed decisions and run the answer-review finalizer before use.',
        ],
        'isolation':'These files are review-only and must never enter training, routing rules, decoding changes, or model selection.',
        'warning':'Assignment is not completion. No decision, rating, time, signature, or independence attestation was pre-filled.',
    }
    (output/'assignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    guide=(
        f'# {benchmark_id} 独立人工审核任务\n\n'
        f'指定审核编号：`{reviewer_id}`\n\n'
        f'共 {len(candidates)} 条，分为 {len(batches)} 批，每批最多 {batch_size} 条。\n\n'
        '逐条同时打开同一批次的 `candidates.jsonl` 和 `decisions.pending.jsonl`。'
        '审核人必须亲自填写决定、五项评分、实际审核时间、签名及独立性声明。\n\n'
        '不得批量默认通过，也不得把候选答案或评分要求交给训练、路由或解码调参流程。'
        '全部批次完成后，合并决定文件并运行答案审核封存校验。\n'
    )
    (output/'REVIEW_ASSIGNMENT.md').write_text(guide,encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reviewer-id',required=True)
    parser.add_argument('--batch-size',type=int,default=50)
    args=parser.parse_args()
    report=create_assignment(args.package,args.output,args.reviewer_id,args.batch_size)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
