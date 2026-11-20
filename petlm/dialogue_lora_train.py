"""Train a standalone LoRA behavior adapter on a frozen PetLM checkpoint."""
import argparse
import json
from pathlib import Path
import random
import time

import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .data import BASE
from .dialogue_train import (evaluate, load_pairs, load_training_pairs, load_validation_groups,
                             make_batch)
from .lora import adapter_state_dict, inject_lora, trainable_parameter_count
from .model import PetLM
from .run import checksum


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for LoRA training.')
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new directory.')
    manifest_path=args.data/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    capability=manifest.get('capability')
    base=torch.load(args.base,map_location='cpu',weights_only=True)
    tokenizer_hash=checksum(args.tokenizer)
    if tokenizer_hash!=manifest['tokenizer_sha256'] or tokenizer_hash!=base['tokenizer_sha256']:
        raise RuntimeError('Tokenizer, data and base checkpoint do not match.')
    tokenizer=spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    single_group=manifest.get('single_group_training') is True
    if single_group:
        training=load_pairs(args.data/'train.jsonl',tokenizer,base['config']['context'])
        natural_indices=[]
        behavior_indices=list(range(len(training)))
        validation_groups={'behavior':load_pairs(
            args.data/'validation.jsonl',tokenizer,base['config']['context'])}
        if args.natural_fraction != 0:
            raise RuntimeError('Development-only single-group training requires natural_fraction=0.')
    else:
        training,natural_indices,behavior_indices=load_training_pairs(
            args.data/'train.jsonl',tokenizer,base['config']['context'])
        if not natural_indices or not behavior_indices:
            raise RuntimeError('LoRA training requires natural and behavior rows.')
        validation_groups=load_validation_groups(
            args.data/'validation.jsonl',tokenizer,base['config']['context'])
    device=torch.device('cuda')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=PetLM(base['config']).to(device)
    model.load_state_dict(base['model'])
    targets=inject_lora(model,args.rank,args.alpha,args.dropout)
    trainable=trainable_parameter_count(model)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=args.learning_rate,weight_decay=0.0)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    rng=random.Random(args.seed)
    args.output.mkdir(parents=True)
    base_hash=checksum(args.base)
    best=float('inf')
    stale=0
    history=[]

    def grouped_evaluation():
        values={name:evaluate(model,pairs,args.batch_size,device,dtype)
                for name,pairs in validation_groups.items()}
        score=(values['behavior'] if single_group else
               args.behavior_validation_weight*values['behavior']+
               (1-args.behavior_validation_weight)*values['natural'])
        return score,values

    def save(name,step,score):
        state={'kind':'petlm_lora_adapter_v1','adapter_state':adapter_state_dict(model),
               'lora':{'rank':args.rank,'alpha':args.alpha,'dropout':args.dropout,
                       'targets':targets},'config':base['config'],'step':step,
               'score':score,'tokenizer_sha256':tokenizer_hash,
               'base_checkpoint':str(args.base.resolve()),
               'base_checkpoint_sha256':base_hash,'dataset_manifest_sha256':checksum(manifest_path),
               'training_group':'behavior_adapter','capability':capability}
        torch.save(state,args.output/f'{name}.pt')

    baseline,baseline_groups=grouped_evaluation()
    best=baseline
    save('best',0,baseline)
    save('latest',0,baseline)
    group_text=', '.join(f'{name}={value:.4f}' for name,value in baseline_groups.items())
    print(f'Base validation={baseline:.4f} ({group_text})',flush=True)
    started=time.perf_counter()
    completed=0
    with SummaryWriter(str(args.output/'runs')) as writer:
        for name,value in baseline_groups.items():
            writer.add_scalar(f'lora/validation_{name}_loss',value,0)
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum=0.0
            for _ in range(args.accumulation):
                indices=[rng.choice(natural_indices if natural_indices and rng.random()<args.natural_fraction
                                    else behavior_indices) for _ in range(args.batch_size)]
                x,y=make_batch(training,indices,device)
                with torch.autocast('cuda',dtype=dtype):
                    logits=model(x)
                    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                                         y.reshape(-1),ignore_index=-100)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite LoRA loss.')
                scaler.scale(loss/args.accumulation).backward()
                loss_sum+=loss.item()/args.accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
            scaler.step(optimizer)
            scaler.update()
            completed=step+1
            writer.add_scalar('lora/train_loss',loss_sum,completed)
            if completed==1 or completed%10==0:
                print(f'step={completed} lora_loss={loss_sum:.4f}',flush=True)
            if completed%args.eval_every==0 or completed==args.steps:
                score,groups=grouped_evaluation()
                history.append({'step':completed,'train_loss':loss_sum,'validation_loss':score,
                                'validation_by_group':groups})
                for name,value in groups.items():
                    writer.add_scalar(f'lora/validation_{name}_loss',value,completed)
                if score<best:
                    best=score
                    stale=0
                    save('best',completed,score)
                else:
                    stale+=1
                save('latest',completed,score)
                group_text=', '.join(f'{name}={value:.4f}' for name,value in groups.items())
                print(f'validation={score:.4f}, {group_text}, best={best:.4f}',flush=True)
                if stale>=args.patience:
                    print('Stopped after repeated validation regressions.',flush=True)
                    break
    report={'kind':'petlm_lora_training_v1','base_checkpoint':str(args.base.resolve()),
            'base_checkpoint_sha256':base_hash,'capability':capability,'training_pairs':len(training),
            'natural_training_pairs':len(natural_indices),'behavior_training_pairs':len(behavior_indices),
            'trainable_parameters':trainable,'total_base_parameters':sum(p.numel() for p in model.parameters())-trainable,
            'lora':{'rank':args.rank,'alpha':args.alpha,'dropout':args.dropout,'targets':targets},
            'natural_fraction':args.natural_fraction,
            'single_group_training':single_group,
            'behavior_validation_weight':args.behavior_validation_weight,
            'baseline_validation_loss':baseline,'baseline_validation_by_group':baseline_groups,
            'best_validation_loss':best,'completed_steps':completed,'target_steps':args.steps,
            'seconds':time.perf_counter()-started,'history':history,'dataset':manifest}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('dataset','history','lora')},
                     ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--alpha',type=float,default=16.0)
    parser.add_argument('--dropout',type=float,default=0.05)
    parser.add_argument('--steps',type=int,default=160)
    parser.add_argument('--learning-rate',type=float,default=0.0002)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--eval-every',type=int,default=20)
    parser.add_argument('--patience',type=int,default=4)
    parser.add_argument('--natural-fraction',type=float,default=0.2)
    parser.add_argument('--behavior-validation-weight',type=float,default=0.8)
    parser.add_argument('--seed',type=int,default=20261111)
    args=parser.parse_args()
    if not 0<=args.natural_fraction<=1 or not 0<=args.behavior_validation_weight<=1:
        parser.error('Fractions must be between zero and one.')
    train(args)


if __name__=='__main__':
    main()
