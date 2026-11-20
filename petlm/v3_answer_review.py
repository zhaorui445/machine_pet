"""Prepare an auditable, item-level review package for answer references."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .run import checksum


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open('w',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def compact(text):
    return ''.join(text.split())


def build(benchmark, output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    benchmark_manifest=json.loads((benchmark/'manifest.json').read_text(encoding='utf-8'))
    prompts={row['id']:row for row in read_jsonl(benchmark/'prompts.jsonl')}
    key_path=benchmark/'answer-key.generated.jsonl'
    if not key_path.exists():
        key_path=benchmark/'answer-key.pending-review.jsonl'
    keys=read_jsonl(key_path)
    if set(prompts)!={row['id'] for row in keys}:
        raise RuntimeError('Prompt and answer-key ids differ.')
    candidates=[]
    decisions=[]
    statuses=Counter()
    for key in keys:
        answer=compact(key['answer'])
        requirements=[any(compact(option) in answer for option in group)
                      for group in key['requirements']]
        forbidden=[word for word in key['forbidden'] if compact(word) in answer]
        public=prompts[key['id']]
        expected_questions=key.get('question_mark_count')
        actual_questions=key['answer'].count('？')+key['answer'].count('?')
        question_count_ok=(expected_questions is None or actual_questions==expected_questions)
        structural=(all(requirements) and not forbidden and question_count_ok and
                    key['capability']==public['capability'] and bool(answer))
        status='automatic_consistency_pass' if structural else 'automatic_consistency_fail'
        statuses[status]+=1
        candidates.append({
            'id':key['id'],'capability':key['capability'],'template_id':key['template_id'],
            'history':public['history'],'user':public['user'],'reference_answer':key['answer'],
            'requirements':key['requirements'],'forbidden':key['forbidden'],
            'automatic_checks':{'requirement_groups':requirements,'forbidden_hits':forbidden,
                                'question_mark_count':actual_questions,
                                'question_count_ok':question_count_ok,
                                'capability_matches':key['capability']==public['capability'],
                                'status':status},
        })
        decisions.append({
            'id':key['id'],'decision':'pending','factuality':'','safety':'',
            'age_suitability':'','naturalness':'','requirements_complete':'',
            'notes':'','reviewer_id':'','reviewed_at':'','reviewer_signature':'',
            'independent_of_generator':False,
        })
    output.mkdir(parents=True)
    write_jsonl(output/'answer-review-candidates.jsonl',candidates)
    write_jsonl(output/'review-decisions.template.jsonl',decisions)
    report={
        'kind':'answer_key_review_package_v2',
        'benchmark_id':benchmark_manifest.get('benchmark_id','unknown'),
        'status':'pending_independent_human_review',
        'count':len(keys),'automatic_consistency':dict(statuses),
        'benchmark':str(benchmark),'benchmark_manifest_sha256':checksum(benchmark/'manifest.json'),
        'prompts_sha256':checksum(benchmark/'prompts.jsonl'),
        'source_answer_key':str(key_path),'source_answer_key_sha256':checksum(key_path),
        'candidates_sha256':checksum(output/'answer-review-candidates.jsonl'),
        'review_template_sha256':checksum(output/'review-decisions.template.jsonl'),
        'warning':'Automatic consistency checks are not independent human review. '
                  'Every reference must be read and approved or rejected by a real reviewer.'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    guide=(
        f'# {report["benchmark_id"]} 参考答案逐条审核\n\n'
        f'当前 {len(keys)} 条参考答案只通过了程序结构检查，尚未完成人工审核。\n\n'
        '请逐条对照 `answer-review-candidates.jsonl`，在审核决定文件中填写决定、'
        '事实性、安全性、年龄适配、自然度、要求完整性、审核说明、审核人编号、'
        '实际审核时间、签名和独立性声明。\n\n'
        '不能批量默认通过。全部条目完成独立逐条审核并通过哈希校验前，'
        '不得生成候选输出、选择模型、调整参数或进行正式评分。\n'
    )
    (output/'REVIEW_GUIDE.md').write_text(guide,encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.benchmark,args.output)


if __name__=='__main__':
    main()
