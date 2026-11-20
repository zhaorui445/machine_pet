"""Measure whether a proposed architecture can complete one optimizer step."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from petlm.model import PetLM


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--sequence',type=int,default=1024)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.sequence>config['context']:
        raise ValueError('Sequence exceeds configured context.')
    device=torch.device('cuda')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model=PetLM(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    ids=torch.randint(0,config['vocab_size'],(config['batch_size'],args.sequence),device=device)
    targets=torch.randint(0,config['vocab_size'],ids.shape,device=device)
    with torch.autocast('cuda',dtype=dtype):
        logits=model(ids)
        loss=F.cross_entropy(logits.reshape(-1,config['vocab_size']),targets.reshape(-1))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    report={'parameters':sum(parameter.numel() for parameter in model.parameters()),
        'context':config['context'],'tested_sequence':args.sequence,'batch_size':config['batch_size'],
        'precision':str(dtype),'loss':loss.item(),
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'status':'one_full_optimizer_step_completed'}
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
