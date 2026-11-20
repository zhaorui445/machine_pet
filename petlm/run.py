"""Train a tokenizer and a causal language model from random weights."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .data import BASE, ROOT
from .model import PetLM
from .window_sampling import validate_schedule, step_offsets, window_order


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def tokenize(args):
    prepared = args.data
    target = args.tokens
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Token dataset exists; choose a new --tokens directory.')
    for split in ('train','validation','test'):
        if not (prepared/f'{split}.jsonl').is_file():
            raise FileNotFoundError(prepared/f'{split}.jsonl')
    target.mkdir(parents=True,exist_ok=True)

    def sentences():
        with (prepared/'train.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                yield json.loads(line)['text'].replace('\n',' ')

    existing = getattr(args,'tokenizer_from',None)
    byte_fallback = getattr(args,'byte_fallback',True)
    character_coverage = getattr(args,'character_coverage',0.9995)
    if existing:
        model_bytes = existing.read_bytes()
        parent_manifest_path = existing.parent / 'manifest.json'
        parent_manifest = json.loads(parent_manifest_path.read_text(encoding='utf-8'))
        if parent_manifest['tokenizer_sha256'] != hashlib.sha256(model_bytes).hexdigest():
            raise RuntimeError('Reused tokenizer differs from its parent manifest.')
        tokenizer_training = parent_manifest['tokenizer_training']
    else:
        # SentencePiece sees training data only; held-out splits are never used to fit it.
        model_buffer = io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=sentences(),model_writer=model_buffer,
            vocab_size=args.vocab_size,model_type='bpe',character_coverage=character_coverage,
            byte_fallback=byte_fallback,
            normalization_rule_name='identity',
            hard_vocab_limit=False,bos_id=1,eos_id=2,unk_id=0,pad_id=-1,
            max_sentence_length=800000,input_sentence_size=100000,shuffle_input_sentence=True)
        model_bytes = model_buffer.getvalue()
        tokenizer_training = {'model_type':'bpe','byte_fallback':byte_fallback,
                              'character_coverage':character_coverage}
    # Python file I/O avoids native SentencePiece path issues with Chinese Windows paths.
    (target/'tokenizer.model').write_bytes(model_bytes)
    tokenizer = spm.SentencePieceProcessor(model_proto=model_bytes)
    counts = {}
    hashes = {}
    for split in ('train','validation','test'):
        count = 0
        source = prepared/f'{split}.jsonl'
        with source.open(encoding='utf-8') as handle, (target/f'{split}.bin').open('wb') as output:
            for line in handle:
                ids = tokenizer.encode(json.loads(line)['text'])+[tokenizer.eos_id()]
                np.asarray(ids,dtype='<u4').tofile(output)
                count += len(ids)
        counts[split] = count
        hashes[split] = checksum(source)
    manifest = {'vocab_size':tokenizer.vocab_size(),'tokens':counts,'input_sha256':hashes,
                'tokenizer_reused_from':str(existing.resolve()) if existing else None,
                'tokenizer_training':tokenizer_training,
                'tokenizer_sha256':checksum(target/'tokenizer.model'),
                'binary_sha256':{s:checksum(target/f'{s}.bin') for s in counts},
                'sources':json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))}
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


def batch(data, size, context, rng, device):
    offsets = rng.integers(0,len(data)-context,size=size)
    return batch_at(data, offsets, context, device)


def batch_at(data, offsets, context, device):
    rows = np.stack([np.array(data[i:i+context+1],dtype=np.int64) for i in offsets])
    rows = torch.from_numpy(rows).to(device)
    return rows[:,:-1],rows[:,1:]


def continuation_state(saved, config, dataset_hash, tokenizer_hash):
    """Validate a new training phase against the saved model and data."""
    if saved['dataset_hash'] != dataset_hash or saved['tokenizer_sha256'] != tokenizer_hash:
        raise RuntimeError('Continuation requires the same tokenized dataset and tokenizer.')
    architecture = ('vocab_size', 'width', 'layers', 'heads', 'context')
    if any(saved['config'][key] != config[key] for key in architecture):
        raise RuntimeError('Continuation cannot change the model architecture.')
    if any(saved['config'][key] != config[key] for key in ('batch_size', 'accumulation', 'seed')):
        raise RuntimeError('Continuation requires the same batch, accumulation and seed settings.')
    if config['steps'] <= saved['step']:
        raise RuntimeError('New total steps must exceed the saved step.')
    return saved['step']


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for training; run check_gpu.py first.')
    manifest_path = args.tokens/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if checksum(args.tokens/'tokenizer.model') != manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer changed after dataset preparation.')
    for split in ('train','validation','test'):
        if checksum(args.tokens/f'{split}.bin') != manifest['binary_sha256'][split]:
            raise RuntimeError(f'{split} token data changed; create a new dataset snapshot.')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config['vocab_size'] = manifest['vocab_size']
    if args.steps is not None:
        config['steps'] = args.steps
    if getattr(args, 'learning_rate', None) is not None:
        config['learning_rate'] = args.learning_rate
    if config['steps'] < 1 or config['learning_rate'] <= 0:
        raise ValueError('Steps and learning rate must be positive.')
    extending = getattr(args, 'extend_from', None)
    adapting = getattr(args, 'adapt_from', None)
    if sum(bool(choice) for choice in (extending, adapting, args.resume)) > 1:
        raise ValueError('Use only one of --extend-from, --adapt-from, or --resume.')
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    train_data = np.memmap(args.tokens/'train.bin',mode='r',dtype='<u4')
    validation = np.memmap(args.tokens/'validation.bin',mode='r',dtype='<u4')
    if min(len(train_data),len(validation)) <= config['context']:
        raise RuntimeError('Insufficient training/validation data. Import more documents.')
    train_order = validate_schedule(len(train_data), config, bool(extending))
    validation_mode = config.get('validation_sampling', 'random_windows')
    if validation_mode not in ('random_windows', 'all_nonoverlapping_windows'):
        raise ValueError('Unknown validation sampling mode.')
    validation_order = window_order(len(validation), config['context']) if validation_mode == 'all_nonoverlapping_windows' else None
    if args.output.exists() and not args.resume:
        raise FileExistsError('Output directory exists; use --resume or a new --output.')
    if (extending or adapting) and args.output.resolve() == (extending or adapting).parent.resolve():
        raise ValueError('Use a new output directory to preserve the original checkpoint.')
    args.output.mkdir(parents=True,exist_ok=True)
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    model = PetLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=0.1)
    dataset_hash = checksum(manifest_path)
    start_step = 0
    phase_start = 0
    parent_checkpoint = None
    parent_step = None
    if args.resume or extending:
        checkpoint = args.resume or extending
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if extending:
            start_step = continuation_state(saved, config, dataset_hash, manifest['tokenizer_sha256'])
            phase_start = start_step
            parent_checkpoint = str(checkpoint.resolve())
            parent_step = saved['step']
        else:
            if saved['config'] != config or saved['dataset_hash'] != dataset_hash:
                raise RuntimeError('Resume requires identical config and dataset, including total schedule steps.')
            start_step = saved['step']
            phase_start = saved.get('phase_start', 0)
            parent_checkpoint = saved.get('parent_checkpoint')
            parent_step = saved.get('parent_step')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    elif adapting:
        saved = torch.load(adapting,map_location='cpu',weights_only=True)
        if saved['tokenizer_sha256'] != manifest['tokenizer_sha256']:
            raise RuntimeError('Adaptation requires the exact same tokenizer as the parent model.')
        architecture = ('vocab_size','width','layers','heads','context')
        if any(saved['config'][key] != config[key] for key in architecture):
            raise RuntimeError('Adaptation requires the same model architecture.')
        model.load_state_dict(saved['model'])
        parent_checkpoint = str(adapting.resolve())
        parent_step = saved['step']
    if start_step >= config['steps']:
        print('Training schedule already complete; existing report preserved. Use --extend-from with a new output and larger --steps.')
        return
    parameters = sum(p.numel() for p in model.parameters())
    print(f'Model: {parameters:,} parameters; precision={dtype}; starting step={start_step}',flush=True)
    print('Sources:',manifest['sources'].get('sources',{}),flush=True)
    print('This is base text pretraining, not instruction/chat training.',flush=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    history = []
    completed = start_step

    def save(step):
        state = {'config':config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scaler':scaler.state_dict(),'step':step,'dataset_hash':dataset_hash,
                 'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all(),'tokenizer_sha256':manifest['tokenizer_sha256'],
                 'phase_start':phase_start,'parent_checkpoint':parent_checkpoint,
                 'parent_step':parent_step}
        temporary = args.output/'latest.tmp'
        torch.save(state,temporary)
        temporary.replace(args.output/'latest.pt')

    with SummaryWriter(str(args.output/'runs')) as writer:
        for index in range(start_step,config['steps']):
            model.train()
            phase_length = config['steps']-phase_start
            warmup = 0 if phase_start else min(config['warmup_steps'],max(1,phase_length//10))
            progress = min(1,max(0,(index-phase_start-warmup)/max(1,phase_length-warmup)))
            warmup_factor = min(1,(index-phase_start+1)/warmup) if warmup else 1
            lr = config['learning_rate'] * warmup_factor * (0.1+0.9*0.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for microstep in range(config['accumulation']):
                if train_order is None:
                    x,y = batch(train_data,config['batch_size'],config['context'],rng,device)
                else:
                    offsets = step_offsets(train_order,index,microstep,config['batch_size'],config['accumulation'])
                    x,y = batch_at(train_data,offsets,config['context'],device)
                with torch.autocast('cuda',dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1,config['vocab_size']),y.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; use last successful checkpoint.')
                train_loss += loss.item()/config['accumulation']
                scaler.scale(loss/config['accumulation']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            completed = index+1
            writer.add_scalar('loss/train',train_loss,completed)
            if completed == 1 or completed % 10 == 0:
                print(f'step={completed} train_loss={train_loss:.4f}',flush=True)
            if completed % config['eval_every'] == 0 or completed == config['steps']:
                model.eval()
                losses, loss_weights = [], []
                eval_rng = np.random.default_rng(12345)
                with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
                    eval_batches = (range(10) if validation_order is None else range(0,len(validation_order),config['batch_size']))
                    for start in eval_batches:
                        if validation_order is None:
                            x,y = batch(validation,config['batch_size'],config['context'],eval_rng,device)
                        else:
                            x,y = batch_at(validation,validation_order[start:start+config['batch_size']],config['context'],device)
                        losses.append(F.cross_entropy(model(x).reshape(-1,config['vocab_size']),y.reshape(-1)).item())
                        loss_weights.append(y.numel())
                val_loss = sum(loss*weight for loss,weight in zip(losses,loss_weights))/sum(loss_weights)
                writer.add_scalar('loss/validation',val_loss,completed)
                history.append({'step':completed,'train_loss':train_loss,'validation_loss':val_loss})
                save(completed)
                print(f'validation_loss={val_loss:.4f}; checkpoint saved',flush=True)
            if args.max_seconds and time.perf_counter()-started >= args.max_seconds:
                save(completed)
                print('Time limit reached; checkpoint saved for continuation.',flush=True)
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    report = {'parameters':parameters,'completed_steps':completed,'target_steps':config['steps'],
        'sampling':config.get('sampling','random_windows'),
        'validation_sampling':validation_mode,
        'unique_training_target_positions_this_run':((completed-start_step)*config['batch_size']*config['context']*config['accumulation'] if train_order is not None else None),
        'phase_start':phase_start,'parent_checkpoint':parent_checkpoint,'parent_step':parent_step,
        'seconds':elapsed,'processed_tokens_this_run':(completed-start_step)*config['batch_size']*config['context']*config['accumulation'],
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'history':history,
        'sources':manifest['sources'],'status':'schedule_complete' if completed==config['steps'] else 'time_limited',
        'warning':'Training completion does not imply useful dialogue or factual accuracy.'}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def generate(args):
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checksum(args.tokens/'tokenizer.model') != state['tokenizer_sha256']:
        raise RuntimeError('Tokenizer does not match checkpoint.')
    tokenizer = spm.SentencePieceProcessor(model_proto=(args.tokens/'tokenizer.model').read_bytes())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PetLM(state['config']).to(device).eval()
    model.load_state_dict(state['model'])
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    ids = tokenizer.encode(args.prompt)
    forbidden = {tokenizer.unk_id(),tokenizer.bos_id()}
    forbidden.update(i for i in range(tokenizer.vocab_size())
                     if tokenizer.id_to_piece(i).startswith('<0x'))
    if not ids:
        raise ValueError('Prompt must not be empty.')
    with torch.no_grad():
        for _ in range(args.new_tokens):
            x = torch.tensor([ids[-state['config']['context']:]],device=device)
            logits = model(x)[0,-1]/0.8
            logits[list(forbidden)] = float('-inf')
            values,indices = logits.topk(min(40,len(logits)))
            token = indices[torch.multinomial(values.softmax(-1),1)].item()
            if token == tokenizer.eos_id():
                break
            ids.append(token)
    print(tokenizer.decode(ids))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command',required=True)
    tok = sub.add_parser('tokenize')
    tok.add_argument('--data',type=Path,default=BASE/'prepared')
    tok.add_argument('--tokens',type=Path,default=BASE/'tokens')
    tok.add_argument('--vocab-size',type=int,default=12000)
    tok.add_argument('--byte-fallback',action=argparse.BooleanOptionalAction,default=True,
        help='Represent unseen characters as bytes; disable for a strictly screened Chinese corpus')
    tok.add_argument('--character-coverage',type=float,default=0.9995)
    tok.add_argument('--tokenizer-from',type=Path,
        help='Reuse a parent model tokenizer when adapting weights to a new corpus')
    tr = sub.add_parser('train')
    tr.add_argument('--config',type=Path,default=ROOT/'configs/pet-small.json')
    tr.add_argument('--tokens',type=Path,default=BASE/'tokens')
    tr.add_argument('--output',type=Path,default=BASE/'models/pet-small')
    tr.add_argument('--steps',type=int)
    tr.add_argument('--learning-rate',type=float)
    tr.add_argument('--max-seconds',type=int,default=600)
    tr.add_argument('--resume',type=Path)
    tr.add_argument('--extend-from',type=Path)
    tr.add_argument('--adapt-from',type=Path,
        help='Initialize from parent weights on a new dataset; reset optimizer and step counter')
    gen = sub.add_parser('generate')
    gen.add_argument('--tokens',type=Path,default=BASE/'tokens')
    gen.add_argument('--checkpoint',type=Path,default=BASE/'models/pet-small/latest.pt')
    gen.add_argument('--prompt',required=True)
    gen.add_argument('--new-tokens',type=int,default=100)
    gen.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.command=='tokenize' and not 0.98<=args.character_coverage<=1.0:
        parser.error('--character-coverage must be between 0.98 and 1.0')
    {'tokenize':tokenize,'train':train,'generate':generate}[args.command](args)


if __name__ == '__main__':
    main()
