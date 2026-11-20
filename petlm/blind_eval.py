"""Run the local model against a keyless, held-out multi-turn prompt file."""
import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch

from .data import BASE
from .dialogue_chat import FALLBACK_REPLY, protected_reply, reply
from .model import PetLM
from .lora import apply_adapter_checkpoint
from .behavior_state import BehaviorController
from .run import checksum


def load_blind_prompts(path):
    rows=[]
    seen=set()
    with path.open(encoding='utf-8') as handle:
        for line_number,line in enumerate(handle,1):
            item=json.loads(line)
            if 'answer' in item or 'rubric' in item:
                raise ValueError(f'Line {line_number}: blind prompt file contains answer material.')
            if item.get('id') in seen:
                raise ValueError(f'Line {line_number}: duplicate id.')
            if not isinstance(item.get('history'),list) or not item.get('user'):
                raise ValueError(f'Line {line_number}: history and user are required.')
            seen.add(item['id'])
            rows.append(item)
    if not rows:
        raise RuntimeError('Blind prompt file is empty.')
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prompts',type=Path,default=BASE/'dialogue'/'expansion-v2'/'blind'/'prompts.jsonl')
    parser.add_argument('--checkpoint',type=Path,default=BASE/'models'/'dialogue-chinese-v5'/'best.pt')
    parser.add_argument('--adapter',type=Path,help='Optional standalone LoRA adapter checkpoint.')
    parser.add_argument('--tokenizer',type=Path,default=BASE/'tokens-chinese-v5'/'tokenizer.model')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('application','model'),default='model')
    parser.add_argument('--capability',help='Evaluate only one capability partition when present.')
    parser.add_argument('--limit',type=int,default=0,help='0 evaluates every prompt.')
    parser.add_argument('--temperature',type=float,default=0.3)
    parser.add_argument('--max-new-tokens',type=int,default=80)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new output file.')
    rows=load_blind_prompts(args.prompts)
    if args.limit<0:
        raise ValueError('Limit cannot be negative.')
    if args.limit:
        rows=rows[:args.limit]
    if args.capability:
        rows=[row for row in rows if row.get('capability')==args.capability]
        if not rows:
            raise RuntimeError(f'No blind prompts found for capability: {args.capability}')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checksum(args.tokenizer)!=state['tokenizer_sha256']:
        raise RuntimeError('Tokenizer does not match checkpoint.')
    tokenizer=spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=PetLM(state['config']).to(device).eval()
    model.load_state_dict(state['model'])
    adapter_hash=None
    if args.adapter:
        adapter=torch.load(args.adapter,map_location='cpu',weights_only=True)
        apply_adapter_checkpoint(model,adapter,checksum(args.checkpoint))
        model.to(device).eval()
        adapter_hash=checksum(args.adapter)
    results=[]
    use_protection=args.mode=='application'
    for index,item in enumerate(rows):
        torch.manual_seed(args.seed+index)
        if device=='cuda':
            torch.cuda.manual_seed_all(args.seed+index)
        history=[(turn['user'],turn['assistant']) for turn in item['history']]
        controller=BehaviorController() if use_protection else None
        answer=reply(model,tokenizer,history,item['user'],args.max_new_tokens,
                     args.temperature,use_protection=use_protection,controller=controller)
        if controller is not None and controller.last_source is not None:
            source=controller.last_source
        elif use_protection and protected_reply(history,item['user']) is not None:
            source='deterministic_rule'
        elif answer==FALLBACK_REPLY:
            source='generation_fallback'
        else:
            source='model_generation'
        results.append({'id':item['id'],'topic':item.get('topic',''),
                        'capability':item.get('capability'),
                        'answer':answer,'response_source':source})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8') as handle:
        for item in results:
            handle.write(json.dumps(item,ensure_ascii=False)+'\n')
    report={'mode':args.mode,'prompts':str(args.prompts),'prompts_sha256':checksum(args.prompts),
            'checkpoint':str(args.checkpoint),'checkpoint_sha256':checksum(args.checkpoint),
            'adapter':str(args.adapter) if args.adapter else None,'adapter_sha256':adapter_hash,
            'capability':args.capability,
            'tokenizer_sha256':checksum(args.tokenizer),'count':len(results),'seed':args.seed,
            'temperature':args.temperature,'output':str(args.output),'output_sha256':checksum(args.output),
            'note':'Keyless generation only; score later against the separately reviewed answer key and rubrics.'}
    manifest=args.output.with_suffix(args.output.suffix+'.manifest.json')
    manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
