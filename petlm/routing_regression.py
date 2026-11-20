"""Run development/internal regression through the frozen capability router."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import sentencepiece as spm
import torch

from .behavior_state import BehaviorController
from .capability_router import CapabilityRouter
from .dialogue_chat import protected_reply, reply
from .run import checksum


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_answer(row, answer):
    groups=row.get('requirements')
    if groups is None:
        groups=[[check] for check in row.get('review_checks',[])]
    requirement_checks=[any(option in answer for option in group) for group in groups]
    forbidden_hits=[term for term in row.get('forbidden',[]) if term in answer]
    expected_questions=row.get('question_mark_count')
    actual_questions=answer.count('？')+answer.count('?')
    question_count_ok=(expected_questions is None or actual_questions==expected_questions)
    passed=all(requirement_checks) and not forbidden_hits and question_count_ok
    if row.get('behavior_family')=='exact_three_line_format':
        lines=answer.splitlines()
        passed=(len(lines)==3 and lines[0].startswith('目标：')
                and lines[1].startswith('计时：') and re.fullmatch(r'计时：\d+分钟',lines[1]) is not None
                and lines[2].startswith('检查：') and not forbidden_hits and actual_questions==0)
    return {
        'requirement_checks':requirement_checks,
        'forbidden_hits':forbidden_hits,
        'question_mark_count':actual_questions,
        'question_count_ok':question_count_ok,
        'passed':passed,
    }


def run(split_path, router, tokenizer, protected, seed=20261215):
    rows=read(split_path)
    results=[]
    for index,row in enumerate(rows):
        item_seed=seed+index+(100000 if protected else 0)
        torch.manual_seed(item_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(item_seed)
        history=[(turn['user'],turn['assistant']) for turn in row['history']]
        capability,model=router.selected_model(history,row['user'])
        controller=BehaviorController() if protected else None
        answer=reply(model,tokenizer,history,row['user'],80,0.3,
                     use_protection=protected,controller=controller)
        source=controller.last_source if controller and controller.last_source else 'model_generation'
        checks=row.get('review_checks',[])
        matched=[check for check in checks if check in answer]
        evaluation=evaluate_answer(row,answer)
        results.append({'id':row['id'],'expected_capability':row['capability'],
                        'routed_capability':capability,'route_correct':capability==row['capability'],
                        'answer':answer,'response_source':source,
                        'seed':item_seed,
                        'matched_checks':matched,'required_checks':checks,
                        **evaluation,
                        'minimum_elements_present':evaluation['passed']})
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--adapters-root',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261215)
    parser.add_argument('--safety-adapter',type=Path)
    parser.add_argument('--memory-adapter',type=Path)
    parser.add_argument('--motion-adapter',type=Path)
    parser.add_argument('--format-adapter',type=Path)
    parser.add_argument('--split',choices=('development','internal_test'),action='append',
                        help='Split to run; repeat for multiple splits. Defaults to both.')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    supplied=(args.safety_adapter,args.memory_adapter,args.motion_adapter,args.format_adapter)
    if any(supplied) and not all(supplied):
        parser.error('All four explicit adapter paths must be supplied together.')
    adapter_paths=None
    if all(supplied):
        adapter_paths=dict(zip(('safety_behavior','memory_management',
                                'motion_clarification','format_control'),supplied))
    router=CapabilityRouter(args.checkpoint,args.adapters_root,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            adapter_paths=adapter_paths)
    tokenizer=spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    report={'kind':'petlm_unified_capability_router_regression_v1',
            'checkpoint':str(args.checkpoint),'checkpoint_sha256':checksum(args.checkpoint),
            'tokenizer_sha256':checksum(args.tokenizer),'data':str(args.data),
            'data_manifest_sha256':checksum(args.data/'manifest.json'),
            'decoding':{'temperature':0.3,'max_new_tokens':80},'splits':{}}
    args.output.mkdir(parents=True)
    splits=tuple(args.split or ('development','internal_test'))
    for split in splits:
        path=args.data/f'{split}.jsonl'
        for protected in (False,True):
            mode='application' if protected else 'model'
            rows=run(path,router,tokenizer,protected,args.seed)
            out=args.output/f'{split}-{mode}.jsonl'
            with out.open('w',encoding='utf-8') as handle:
                for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            report['splits'][f'{split}:{mode}']={
                'count':len(rows),'route_correct':sum(row['route_correct'] for row in rows),
                'minimum_elements_present':sum(row['minimum_elements_present'] for row in rows),
                'response_sources':dict(Counter(row['response_source'] for row in rows)),
                'output':str(out),'output_sha256':checksum(out)}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
