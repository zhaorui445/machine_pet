"""Evaluate assistant reply loss on a held-out dialogue split."""
import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch

from .data import BASE
from .dialogue_train import evaluate, load_pairs
from .model import PetLM
from .run import checksum


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=BASE/'dialogue'/'prepared-oasst1-zh-context')
    parser.add_argument('--pairs',type=Path,
        help='Evaluate an independent JSONL file containing prompt and answer fields.')
    parser.add_argument('--tokenizer',type=Path,default=BASE/'tokens-curated-expanded'/'tokenizer.model')
    parser.add_argument('--split',choices=('validation','test'),default='test')
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    tokenizer_hash=checksum(args.tokenizer)
    if args.pairs:
        if state['tokenizer_sha256']!=tokenizer_hash:
            raise RuntimeError('Tokenizer and dialogue checkpoint do not match.')
        pairs_path=args.pairs
        evaluation_name='independent'
    else:
        manifest=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
        if (tokenizer_hash!=manifest['tokenizer_sha256']
                or state['tokenizer_sha256']!=manifest['tokenizer_sha256']):
            raise RuntimeError('Tokenizer and dialogue dataset do not match.')
        pairs_path=args.data/f'{args.split}.jsonl'
        evaluation_name=args.split
    tokenizer=spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    pairs=load_pairs(pairs_path,tokenizer,state['config']['context'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PetLM(state['config']).to(device).eval()
    model.load_state_dict(state['model'])
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    if device.type=='cpu':
        # The evaluation helper uses CUDA autocast; on CPU it is simpler to require the project's GPU.
        raise RuntimeError('This dialogue evaluation command requires CUDA.')
    loss=evaluate(model,pairs,args.batch_size,device,dtype)
    print(json.dumps({'checkpoint':str(args.checkpoint),'split':evaluation_name,
        'pairs_path':str(pairs_path),'pairs_sha256':checksum(pairs_path),
        'pairs':len(pairs),'reply_token_loss':loss,
        'note':'Reply-only held-out loss; not a measure of conversational reliability.'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
