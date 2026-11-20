"""Two-phase v3 evaluation: freeze keyless outputs, then score them."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

import sentencepiece as spm
import torch

from .behavior_state import BehaviorController
from .capability_router import CapabilityRouter
from .dialogue_chat import FALLBACK_REPLY,protected_reply,reply
from .model import PetLM
from .run import checksum


CODE_FILES=('behavior_state.py','capability_router.py','dialogue_chat.py','model.py',
            'lora.py','frozen_v3_evaluation.py')


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path,rows):
    with path.open('w',encoding='utf-8',newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def load_public_prompts(path):
    rows=read_jsonl(path)
    seen=set()
    for row in rows:
        if 'answer' in row or 'requirements' in row or 'forbidden' in row or 'rubric' in row:
            raise ValueError('Public prompt file contains answer or scoring material.')
        if row.get('id') in seen:
            raise ValueError(f'Duplicate prompt id: {row.get("id")}')
        if not row.get('user') or not isinstance(row.get('history'),list):
            raise ValueError(f'Prompt {row.get("id")} lacks history or user text.')
        seen.add(row['id'])
    if not rows:
        raise RuntimeError('Public prompt file is empty.')
    return rows


def artifact_hashes(checkpoint,tokenizer,adapter_root):
    adapters={
        capability:str(path) for capability,path in {
            'safety_behavior':adapter_root/'dialogue-v33-lora-human-safety-r8'/'best.pt',
            'memory_management':adapter_root/'dialogue-v33-lora-human-memory-r8'/'best.pt',
            'motion_clarification':adapter_root/'dialogue-v33-lora-human-motion-r8'/'best.pt',
            'format_control':adapter_root/'dialogue-v33-lora-human-format-r8'/'best.pt',
        }.items()}
    for path in adapters.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    package=Path(__file__).resolve().parent
    return {
        'checkpoint':{'path':str(checkpoint),'sha256':checksum(checkpoint)},
        'tokenizer':{'path':str(tokenizer),'sha256':checksum(tokenizer)},
        'adapters':{name:{'path':path,'sha256':checksum(Path(path))}
                    for name,path in adapters.items()},
        'code':{name:{'path':str(package/name),'sha256':checksum(package/name)}
                for name in CODE_FILES},
    }


def verify_artifacts(artifacts):
    for group in ('checkpoint','tokenizer'):
        item=artifacts[group]
        if checksum(Path(item['path']))!=item['sha256']:
            raise RuntimeError(f'Frozen {group} hash changed.')
    for group in ('adapters','code'):
        for name,item in artifacts[group].items():
            if checksum(Path(item['path']))!=item['sha256']:
                raise RuntimeError(f'Frozen {group} artifact changed: {name}')


def generate(prompts,benchmark_manifest,checkpoint,adapter_root,tokenizer,output,
             seed,temperature,max_new_tokens):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output directory.')
    public=load_public_prompts(prompts)
    benchmark=json.loads(benchmark_manifest.read_text(encoding='utf-8'))
    if checksum(prompts)!=benchmark['prompts_sha256']:
        raise RuntimeError('Public prompts no longer match the frozen benchmark manifest.')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if checksum(tokenizer)!=state['tokenizer_sha256']:
        raise RuntimeError('Tokenizer does not match the checkpoint.')
    processor=spm.SentencePieceProcessor(model_proto=tokenizer.read_bytes())
    device='cuda' if torch.cuda.is_available() else 'cpu'
    router=CapabilityRouter(checkpoint,adapter_root,device=device)
    base=PetLM(state['config']).to(device).eval()
    base.load_state_dict(state['model'])
    results={'model':[],'application':[]}
    route_errors=[]
    for index,item in enumerate(public):
        history=[(turn['user'],turn['assistant']) for turn in item['history']]
        capability,selected=router.selected_model(history,item['user'])
        if capability!=item.get('capability'):
            route_errors.append({'id':item['id'],'expected':item.get('capability'),'routed':capability})
        model=selected or base
        for mode_offset,mode in enumerate(('model','application')):
            item_seed=seed+(index*2)+mode_offset
            torch.manual_seed(item_seed)
            if device=='cuda':
                torch.cuda.manual_seed_all(item_seed)
            protected=mode=='application'
            controller=BehaviorController() if protected else None
            answer=reply(model,processor,history,item['user'],max_new_tokens,temperature,
                         use_protection=protected,controller=controller)
            if controller is not None and controller.last_source is not None:
                source=controller.last_source
            elif protected and protected_reply(history,item['user']) is not None:
                source='deterministic_rule'
            elif answer==FALLBACK_REPLY:
                source='generation_fallback'
            else:
                source='model_generation'
            results[mode].append({
                'id':item['id'],'template_id':item.get('template_id'),
                'capability':item.get('capability'),'routed_capability':capability,
                'route_correct':capability==item.get('capability'),
                'answer':answer,'response_source':source,'seed':item_seed,
            })
    output.mkdir(parents=True)
    output_paths={}
    for mode,rows in results.items():
        path=output/f'{mode}-outputs.jsonl'
        write_jsonl(path,rows)
        output_paths[mode]={'path':str(path),'sha256':checksum(path),'count':len(rows)}
    artifacts=artifact_hashes(checkpoint,tokenizer,adapter_root)
    report={
        'kind':'petlm_frozen_v3_keyless_generation_v1',
        'phase':'outputs_frozen_before_answer_key_access',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'prompts':str(prompts),'prompts_sha256':checksum(prompts),
        'benchmark_manifest':str(benchmark_manifest),
        'benchmark_manifest_sha256':checksum(benchmark_manifest),
        'benchmark_id':benchmark.get('benchmark_id'),'count':len(public),
        'decoding':{'seed':seed,'temperature':temperature,
                    'max_new_tokens':max_new_tokens},
        'device_type':device,'artifacts':artifacts,'outputs':output_paths,
        'route_errors':route_errors,'route_correct':len(public)-len(route_errors),
        'isolation':('Generation accepted only the public prompt file. No answer key, requirements, '
                     'forbidden terms, or scoring report was accepted or read in this phase.'),
    }
    (output/'generation-manifest.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def score_mode(predictions,keys):
    predicted={row['id']:row for row in predictions}
    if len(predicted)!=len(predictions) or set(predicted)!={row['id'] for row in keys}:
        raise RuntimeError('Frozen outputs and reviewed answer key do not cover identical unique ids.')
    passed=0
    families=defaultdict(Counter)
    capabilities=defaultdict(Counter)
    sources=Counter()
    failures=[]
    for key in keys:
        prediction=predicted[key['id']]
        text=''.join(prediction['answer'].split())
        checks=[any(''.join(option.split()) in text for option in group)
                for group in key['requirements']]
        forbidden=[word for word in key.get('forbidden',[])
                   if ''.join(word.split()) in text]
        expected_questions=key.get('question_mark_count')
        actual_questions=prediction['answer'].count('？')+prediction['answer'].count('?')
        question_count_ok=(expected_questions is None or actual_questions==expected_questions)
        ok=all(checks) and not forbidden and question_count_ok
        for buckets,name in ((families,key['template_id']),(capabilities,key['capability'])):
            buckets[name]['total']+=1
            buckets[name]['passed']+=int(ok)
        sources[prediction.get('response_source','missing')]+=1
        passed+=int(ok)
        if not ok:
            failures.append({'id':key['id'],'template_id':key['template_id'],
                             'capability':key['capability'],'answer':prediction['answer'],
                             'requirement_checks':checks,'forbidden_hits':forbidden,
                             'question_mark_count':actual_questions,
                             'question_count_ok':question_count_ok})
    def summarize(values):
        return {name:{'passed':value['passed'],'total':value['total'],
                      'pass_rate':value['passed']/value['total']}
                for name,value in sorted(values.items())}
    return {'passed':passed,'total':len(keys),'pass_rate':passed/len(keys),
            'response_sources':dict(sources),'by_capability':summarize(capabilities),
            'by_template':summarize(families),'failures':failures}


def score(generation,review_manifest,answer_key,output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable score directory.')
    generated=json.loads((generation/'generation-manifest.json').read_text(encoding='utf-8'))
    if generated.get('phase')!='outputs_frozen_before_answer_key_access':
        raise RuntimeError('Generation manifest does not prove keyless output freezing.')
    verify_artifacts(generated['artifacts'])
    for mode,item in generated['outputs'].items():
        if checksum(Path(item['path']))!=item['sha256']:
            raise RuntimeError(f'Frozen {mode} output hash changed.')
    review=json.loads(review_manifest.read_text(encoding='utf-8'))
    if (review.get('status')!='independent_human_review_complete'
            or review.get('usable_as_frozen_answer_key') is not True):
        raise RuntimeError('Answer key has not completed independent human review.')
    if checksum(answer_key)!=review['reviewed_answer_key_sha256']:
        raise RuntimeError('Reviewed answer key hash does not match its sealed manifest.')
    keys=read_jsonl(answer_key)
    reports={mode:score_mode(read_jsonl(Path(item['path'])),keys)
             for mode,item in generated['outputs'].items()}
    output.mkdir(parents=True)
    for mode,report in reports.items():
        (output/f'{mode}-score.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={
        'kind':'petlm_frozen_formal_evaluation_v2',
        'benchmark_id':generated.get('benchmark_id'),
        'created_at':datetime.now(timezone.utc).isoformat(),
        'generation_manifest':str(generation/'generation-manifest.json'),
        'generation_manifest_sha256':checksum(generation/'generation-manifest.json'),
        'review_manifest':str(review_manifest),'review_manifest_sha256':checksum(review_manifest),
        'answer_key':str(answer_key),'answer_key_sha256':checksum(answer_key),
        'modes':{mode:{key:value for key,value in report.items() if key!='failures'}
                 for mode,report in reports.items()},
        'isolation_verified':True,
        'note':('This is deterministic requirement/forbidden-term scoring against an independently '
                'reviewed frozen key, not a new human quality score.'),
    }
    (output/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    generator=sub.add_parser('generate')
    generator.add_argument('--prompts',type=Path,required=True)
    generator.add_argument('--benchmark-manifest',type=Path,required=True)
    generator.add_argument('--checkpoint',type=Path,required=True)
    generator.add_argument('--adapters-root',type=Path,required=True)
    generator.add_argument('--tokenizer',type=Path,required=True)
    generator.add_argument('--output',type=Path,required=True)
    generator.add_argument('--seed',type=int,default=20261108)
    generator.add_argument('--temperature',type=float,default=0.3)
    generator.add_argument('--max-new-tokens',type=int,default=80)
    scorer=sub.add_parser('score')
    scorer.add_argument('--generation',type=Path,required=True)
    scorer.add_argument('--review-manifest',type=Path,required=True)
    scorer.add_argument('--answer-key',type=Path,required=True)
    scorer.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='generate':
        result=generate(args.prompts,args.benchmark_manifest,args.checkpoint,
                        args.adapters_root,args.tokenizer,args.output,args.seed,
                        args.temperature,args.max_new_tokens)
    else:
        result=score(args.generation,args.review_manifest,args.answer_key,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
