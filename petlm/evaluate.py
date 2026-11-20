"""Compare checkpoints on identical held-out token windows."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import PetLM
from .run import batch, checksum


def evaluate(checkpoint, tokens, split, batches):
    if split not in ('validation','test') or batches < 1:
        raise ValueError('Use a held-out split and a positive number of batches.')
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    manifest = json.loads((tokens/'manifest.json').read_text(encoding='utf-8'))
    if state['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise RuntimeError('Checkpoint and dataset tokenizers differ.')
    path = tokens/f'{split}.bin'
    if checksum(path) != manifest['binary_sha256'][split]:
        raise RuntimeError('Held-out token file changed after preparation.')
    data = np.memmap(path,mode='r',dtype='<u4')
    config = state['config']
    if len(data) <= config['context']:
        raise RuntimeError('Held-out split is too small.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PetLM(config).to(device).eval()
    model.load_state_dict(state['model'])
    rng = np.random.default_rng(12345)
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x,y = batch(data,config['batch_size'],config['context'],rng,device)
            if device.type == 'cuda':
                with torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    logits = model(x)
            else:
                logits = model(x)
            losses.append(F.cross_entropy(logits.float().reshape(-1,config['vocab_size']),y.reshape(-1)).item())
    loss = sum(losses)/len(losses)
    return {'checkpoint':str(checkpoint),'split':split,'batches':batches,
            'sampled_tokens':batches*config['batch_size']*config['context'],
            'loss':loss,'perplexity':math.exp(loss),
            'note':'Same token data and RNG windows enable checkpoint comparison; no inference of chat quality.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--tokens',type=Path,required=True)
    parser.add_argument('--split',choices=('validation','test'),default='validation')
    parser.add_argument('--batches',type=int,default=50)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint,args.tokens,args.split,args.batches),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
