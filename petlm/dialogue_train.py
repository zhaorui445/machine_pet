"""Supervised fine-tuning with loss only on assistant reply tokens."""
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
from .model import PetLM
from .run import checksum


def encode_pair(tokenizer, prompt, answer, context):
    prefix=tokenizer.encode(prompt)
    reply=tokenizer.encode(answer)+[tokenizer.eos_id()]
    if len(prefix)+len(reply)>context:
        raise ValueError('Dialogue pair exceeds the model context.')
    ids=prefix+reply
    targets=ids[1:]
    targets[:len(prefix)-1]=[-100]*(len(prefix)-1)
    return ids[:-1],targets


def load_pairs(path, tokenizer, context):
    pairs=[]
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            pairs.append(encode_pair(tokenizer,row['prompt'],row['answer'],context))
    if not pairs:
        raise RuntimeError(f'No dialogue pairs in {path}')
    return pairs


def _is_behavior_group(group):
    return group in ('diverse_synthetic','confirmed_template',
                     'human_reviewed_behavior_v1','human_approved_behavior_v1',
                     'rule_authored_behavior_hardening_v1',
                     'rule_authored_capability_expansion_v2',
                     'human_reviewed_dialogue_v2') or str(group or '').startswith('human_reviewed_dialogue_')


def load_training_pairs(path, tokenizer, context):
    pairs=[]
    legacy=[]
    augmented=[]
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            index=len(pairs)
            pairs.append(encode_pair(tokenizer,row['prompt'],row['answer'],context))
            if _is_behavior_group(row.get('training_group')):
                augmented.append(index)
            else:
                legacy.append(index)
    if not pairs:
        raise RuntimeError(f'No dialogue pairs in {path}')
    return pairs,legacy,augmented


def load_validation_groups(path, tokenizer, context):
    groups={'natural':[],'behavior':[]}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            group='behavior' if _is_behavior_group(row.get('training_group')) else 'natural'
            groups[group].append(encode_pair(tokenizer,row['prompt'],row['answer'],context))
    if not groups['natural']:
        raise RuntimeError('Grouped validation requires natural examples.')
    return groups


def make_batch(pairs, indices, device):
    length=max(len(pairs[i][0]) for i in indices)
    x=torch.zeros((len(indices),length),dtype=torch.long,device=device)
    y=torch.full((len(indices),length),-100,dtype=torch.long,device=device)
    for row,index in enumerate(indices):
        ids,targets=pairs[index]
        x[row,:len(ids)]=torch.tensor(ids,device=device)
        y[row,:len(targets)]=torch.tensor(targets,device=device)
    return x,y


def evaluate(model,pairs,batch_size,device,dtype):
    model.eval()
    weighted_loss=0.0
    tokens=0
    with torch.no_grad():
        for start in range(0,len(pairs),batch_size):
            x,y=make_batch(pairs,range(start,min(len(pairs),start+batch_size)),device)
            with torch.autocast('cuda',dtype=dtype):
                logits=model(x)
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),
                                     ignore_index=-100,reduction='sum')
            weighted_loss+=loss.item()
            tokens+=(y!=-100).sum().item()
    return weighted_loss/tokens


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for dialogue training.')
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'{args.output} exists; use --resume or choose a new directory.')
    manifest_path=args.data/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    tokenizer_path=args.tokenizer
    if checksum(tokenizer_path)!=manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer differs from dialogue data preparation.')
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    if parent['tokenizer_sha256']!=manifest['tokenizer_sha256']:
        raise RuntimeError('Parent model and dialogue tokenizer differ.')
    config=parent['config']
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    training,legacy_indices,augmented_indices=load_training_pairs(
        args.data/'train.jsonl',tokenizer,config['context'])
    if args.legacy_replay_fraction and (not legacy_indices or not augmented_indices):
        raise RuntimeError('Legacy replay requires both legacy and augmented training rows.')
    validation_path=args.validation or (args.data/'validation.jsonl')
    validation_groups=load_validation_groups(validation_path,tokenizer,config['context'])
    validation=[pair for group in validation_groups.values() for pair in group]
    device=torch.device('cuda')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=PetLM(config).to(device)
    model.load_state_dict(parent['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.01)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    rng=random.Random(args.seed)
    start_step=0
    best_loss=float('inf')
    stale=0
    dataset_hash=checksum(manifest_path)
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if (saved['dataset_hash']!=dataset_hash or saved['parent_checkpoint']!=str(args.parent.resolve())
                or saved['learning_rate']!=args.learning_rate or saved['target_steps']!=args.steps
                or saved.get('legacy_replay_fraction',0.0)!=args.legacy_replay_fraction):
            raise RuntimeError('Resume requires identical data, parent and training settings.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        rng.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_step=saved['step']
        best_loss=saved['best_validation_loss']
        stale=saved['stale_evaluations']
    args.output.mkdir(parents=True,exist_ok=True)

    def save(name,step):
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'config':config,'step':step,'target_steps':args.steps,'learning_rate':args.learning_rate,
            'best_validation_loss':best_loss,'stale_evaluations':stale,
            'dataset_hash':dataset_hash,'tokenizer_sha256':manifest['tokenizer_sha256'],
            'parent_checkpoint':str(args.parent.resolve()),'parent_step':parent['step'],
            'legacy_replay_fraction':args.legacy_replay_fraction,
            'python_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'training_type':'assistant-only SFT'}
        temporary=args.output/f'{name}.tmp'
        torch.save(state,temporary)
        temporary.replace(args.output/f'{name}.pt')

    baseline_groups=({name:evaluate(model,pairs,args.batch_size,device,dtype)
                      for name,pairs in validation_groups.items() if pairs} if not args.resume else None)
    effective_behavior_weight = (args.behavior_validation_weight
                                 if validation_groups.get('behavior') else 0.0)
    baseline=(effective_behavior_weight*baseline_groups.get('behavior', 0.0)+
              (1-effective_behavior_weight)*baseline_groups['natural']) if baseline_groups else None
    if baseline is not None:
        best_loss=baseline
        groups_text=', '.join(f'{name}={value:.4f}' for name,value in baseline_groups.items())
        print(f'Parent grouped validation reply loss={baseline:.4f} ({groups_text})',flush=True)
    history=[]
    started=time.perf_counter()
    completed=start_step
    with SummaryWriter(str(args.output/'runs')) as writer:
        if baseline is not None:
            writer.add_scalar('dialogue/validation_loss',baseline,0)
            for name,value in baseline_groups.items():
                writer.add_scalar(f'dialogue/validation_{name}_loss',value,0)
        for step in range(start_step,args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum=0.0
            for _ in range(args.accumulation):
                if args.legacy_replay_fraction:
                    indices=[rng.choice(legacy_indices if rng.random()<args.legacy_replay_fraction
                                        else augmented_indices) for _ in range(args.batch_size)]
                else:
                    indices=[rng.randrange(len(training)) for _ in range(args.batch_size)]
                x,y=make_batch(training,indices,device)
                with torch.autocast('cuda',dtype=dtype):
                    logits=model(x)
                    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite dialogue loss.')
                scaler.scale(loss/args.accumulation).backward()
                loss_sum+=loss.item()/args.accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            completed=step+1
            writer.add_scalar('dialogue/train_loss',loss_sum,completed)
            if completed==1 or completed%10==0:
                print(f'step={completed} reply_loss={loss_sum:.4f}',flush=True)
            if completed%args.eval_every==0 or completed==args.steps:
                grouped={name:evaluate(model,pairs,args.batch_size,device,dtype)
                         for name,pairs in validation_groups.items() if pairs}
                value=(effective_behavior_weight*grouped.get('behavior', 0.0)+
                       (1-effective_behavior_weight)*grouped['natural'])
                writer.add_scalar('dialogue/validation_loss',value,completed)
                for name,group_loss in grouped.items():
                    writer.add_scalar(f'dialogue/validation_{name}_loss',group_loss,completed)
                history.append({'step':completed,'train_loss':loss_sum,'validation_loss':value,
                                'validation_by_group':grouped})
                if value<best_loss:
                    best_loss=value
                    stale=0
                    save('best',completed)
                else:
                    stale+=1
                save('latest',completed)
                print(f'validation_reply_loss={value:.4f}, '
                      f"behavior={grouped.get('behavior', float('nan')):.4f}, natural={grouped['natural']:.4f}, "
                      f'best={best_loss:.4f}',flush=True)
                if stale>=args.patience:
                    print('Stopped after repeated validation regressions.',flush=True)
                    break
            if args.max_seconds and time.perf_counter()-started>=args.max_seconds:
                save('latest',completed)
                print('Time limit reached; checkpoint saved.',flush=True)
                break
    report={'parent_checkpoint':str(args.parent.resolve()),'parent_step':parent['step'],
        'training_pairs':len(training),'validation_pairs':len(validation),
        'legacy_training_pairs':len(legacy_indices),'augmented_training_pairs':len(augmented_indices),
        'legacy_replay_fraction':args.legacy_replay_fraction,
        'behavior_validation_weight':args.behavior_validation_weight,
        'baseline_validation_loss':baseline,'best_validation_loss':best_loss,
        'baseline_validation_by_group':baseline_groups,
        'completed_steps':completed,'target_steps':args.steps,'seconds':time.perf_counter()-started,
        'history':history,'dataset':manifest,
        'warning':'A low reply loss does not establish coherent, factual, or safe conversation.'}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('dataset','history')},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=BASE/'dialogue'/'prepared-oasst1-zh')
    parser.add_argument('--validation',type=Path,
                        help='Validation JSONL; defaults to data/validation.jsonl. Use development explicitly when test is reserved.')
    parser.add_argument('--tokenizer',type=Path,default=BASE/'tokens-curated-expanded'/'tokenizer.model')
    parser.add_argument('--parent',type=Path,default=BASE/'models'/'curated-expanded'/'latest.pt')
    parser.add_argument('--output',type=Path,default=BASE/'models'/'dialogue-oasst1-zh')
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--learning-rate',type=float,default=0.00005)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--eval-every',type=int,default=50)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--max-seconds',type=int,default=600)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--legacy-replay-fraction',type=float,default=0.0)
    parser.add_argument('--behavior-validation-weight',type=float,default=0.7)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if min(args.steps,args.batch_size,args.accumulation,args.eval_every,args.patience)<1 or args.learning_rate<=0:
        parser.error('Steps, batch size, accumulation, evaluation interval, patience and learning rate must be positive.')
    if not 0<=args.legacy_replay_fraction<1:
        parser.error('legacy-replay-fraction must be at least 0 and below 1.')
    if not 0<=args.behavior_validation_weight<=1:
        parser.error('behavior-validation-weight must be between 0 and 1.')
    train(args)


if __name__=='__main__':
    main()
