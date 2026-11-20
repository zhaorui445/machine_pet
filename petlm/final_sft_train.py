"""Fixed-protocol, capability-stratified LoRA training for the final SFT snapshot."""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import random
import time

import sentencepiece as spm
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .dialogue_train import encode_pair, evaluate, make_batch
from .lora import adapter_state_dict, inject_lora, load_adapter_state, trainable_parameter_count
from .model import PetLM
from .run import checksum


PROTOCOL_KIND = 'petlm_final_sft_protocol_v1'
CHECKPOINT_KIND = 'petlm_final_sft_lora_v1'
GROUP_ORDER = ('natural', 'safety', 'memory', 'format', 'motion')


def read_protocol(path):
    protocol=json.loads(path.read_text(encoding='utf-8'))
    if protocol.get('kind')!=PROTOCOL_KIND or protocol.get('status')!='frozen_before_training':
        raise RuntimeError('The training protocol is not a frozen final-SFT protocol.')
    project_root=(path.parent/protocol['project_root']).resolve()
    return protocol,project_root


def protocol_paths(protocol,project_root):
    return {name:(project_root/item['path']).resolve()
            for name,item in protocol['inputs'].items() if isinstance(item,dict) and 'path' in item}


def validate_protocol(protocol):
    groups=protocol['data']['groups']
    if tuple(groups)!=GROUP_ORDER:
        raise RuntimeError(f'Protocol groups must be ordered as {GROUP_ORDER}.')
    for field in ('training_sampling_weights','validation_scoring_weights'):
        weights=protocol['data'][field]
        if tuple(weights)!=GROUP_ORDER or not math.isclose(sum(weights.values()),1.0,abs_tol=1e-12):
            raise RuntimeError(f'{field} must contain the five fixed groups and sum to one.')
        if any(value<=0 for value in weights.values()):
            raise RuntimeError(f'{field} weights must be positive.')
    training=protocol['training']
    if training['method']!='standalone_lora_on_frozen_base':
        raise RuntimeError('Only standalone LoRA is allowed by this trainer.')
    if training['precision']!='bfloat16':
        raise RuntimeError('This fixed protocol requires bfloat16.')
    if training['effective_batch_size']!=training['batch_size']*training['gradient_accumulation']:
        raise RuntimeError('Effective batch size does not match batch size times accumulation.')
    if not 0<training['warmup_steps']<training['maximum_steps']:
        raise RuntimeError('Warmup steps must be between zero and maximum steps.')
    if training['evaluation_interval_steps']<1 or training['maximum_steps']%training['evaluation_interval_steps']:
        raise RuntimeError('The evaluation interval must divide maximum steps.')
    if protocol['restrictions'].get('parameter_overrides') is not False:
        raise RuntimeError('The protocol must prohibit parameter overrides.')


def verify_inputs(protocol,paths):
    verified={}
    for name in ('manifest','integrity_audit','snapshot_status','train','validation',
                 'base_checkpoint','tokenizer','trainer'):
        path=paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual=checksum(path)
        expected=protocol['inputs'][name]['sha256']
        if actual!=expected:
            raise RuntimeError(f'{name} SHA-256 mismatch: expected {expected}, got {actual}.')
        verified[name]=actual
    snapshot=paths['snapshot_directory']
    allowed=set(protocol['restrictions']['allowed_data_files'])
    unexpected=[item.name for item in snapshot.iterdir() if item.is_file() and item.name not in allowed]
    if unexpected:
        raise RuntimeError(f'Unexpected files in frozen snapshot: {unexpected}')
    for name in ('train','validation'):
        lowered=paths[name].name.lower()
        if any(fragment in lowered for fragment in protocol['restrictions']['forbidden_name_fragments']):
            raise RuntimeError(f'Forbidden data path selected for {name}.')
    manifest=json.loads(paths['manifest'].read_text(encoding='utf-8'))
    audit=json.loads(paths['integrity_audit'].read_text(encoding='utf-8'))
    status=json.loads(paths['snapshot_status'].read_text(encoding='utf-8'))
    if manifest.get('training_started') is not False or manifest.get('independent_test_read_or_used') is not False:
        raise RuntimeError('Snapshot manifest does not preserve the pre-training/test isolation boundary.')
    if audit.get('status')!='pass_ready_for_fixed_training_protocol' or audit.get('training_started') is not False:
        raise RuntimeError('Snapshot integrity audit is not approved for the fixed protocol.')
    if (status.get('status')!='final_sft_snapshot_frozen_training_not_started'
            or status.get('training_started') is not False
            or status.get('independent_test_read_or_used') is not False):
        raise RuntimeError('Snapshot status does not preserve the frozen pre-training boundary.')
    return verified


def load_grouped_pairs(path,tokenizer,context,protocol,split):
    capability_to_group={settings['capability']:group
                         for group,settings in protocol['data']['groups'].items()}
    pools={group:[] for group in GROUP_ORDER}
    rows=0
    with path.open(encoding='utf-8') as handle:
        for line_number,line in enumerate(handle,1):
            row=json.loads(line)
            if row.get('split')!=split:
                raise RuntimeError(f'{path.name}:{line_number} has the wrong split.')
            capability=row.get('capability')
            if capability not in capability_to_group:
                raise RuntimeError(f'{path.name}:{line_number} has unknown capability {capability!r}.')
            pools[capability_to_group[capability]].append(
                encode_pair(tokenizer,row['prompt'],row['answer'],context))
            rows+=1
    expected_rows=protocol['inputs'][split]['rows']
    if rows!=expected_rows:
        raise RuntimeError(f'{split} row count mismatch: expected {expected_rows}, got {rows}.')
    for group,pairs in pools.items():
        expected=protocol['data']['groups'][group][f'{split}_rows']
        if len(pairs)!=expected:
            raise RuntimeError(f'{split}/{group} count mismatch: expected {expected}, got {len(pairs)}.')
    return pools


def choose_group(rng,weights):
    marker=rng.random()
    cumulative=0.0
    for group in GROUP_ORDER:
        cumulative+=weights[group]
        if marker<cumulative:
            return group
    return GROUP_ORDER[-1]


def sample_batch(pools,weights,batch_size,rng):
    selected=[]
    sampled_groups=[]
    for _ in range(batch_size):
        group=choose_group(rng,weights)
        selected.append(rng.choice(pools[group]))
        sampled_groups.append(group)
    return selected,sampled_groups


def make_direct_batch(pairs,device):
    return make_batch(pairs,range(len(pairs)),device)


def weighted_score(values,weights):
    return sum(values[group]*weights[group] for group in GROUP_ORDER)


def learning_rate_for_step(training,step):
    base=training['optimizer']['learning_rate']
    warmup=training['warmup_steps']
    maximum=training['maximum_steps']
    minimum=training['schedule']['minimum_learning_rate_ratio']
    if step<=warmup:
        return base*step/warmup
    progress=(step-warmup)/(maximum-warmup)
    multiplier=minimum+(1.0-minimum)*0.5*(1.0+math.cos(math.pi*progress))
    return base*multiplier


def group_guard_results(values,baseline,limits):
    return {group:values[group]<=baseline[group]*(1.0+limits[group]) for group in GROUP_ORDER}


def create_model(base,training,device):
    model=PetLM(base['config']).to(device)
    parent_state=base.pop('model')
    model.load_state_dict(parent_state)
    del parent_state
    settings=training['lora']
    targets=inject_lora(model,settings['rank'],settings['alpha'],settings['dropout'])
    expected=[f'blocks.{layer}.{suffix}' for layer in range(settings['transformer_layers'])
              for suffix in settings['targets']]
    if targets!=expected:
        raise RuntimeError('Injected LoRA targets differ from the frozen protocol.')
    return model,targets


def prepare(protocol_path,require_gpu=True):
    protocol,project_root=read_protocol(protocol_path)
    validate_protocol(protocol)
    paths=protocol_paths(protocol,project_root)
    verified=verify_inputs(protocol,paths)
    loaded=torch.load(paths['base_checkpoint'],map_location='cpu',weights_only=True)
    if loaded.get('tokenizer_sha256')!=verified['tokenizer'] or loaded.get('step')!=protocol['inputs']['base_checkpoint']['step']:
        raise RuntimeError('Base checkpoint metadata differs from the frozen protocol.')
    base={'config':loaded['config'],'model':loaded['model'],'step':loaded['step']}
    del loaded
    if base['config']['context']<protocol['data']['maximum_encoded_tokens']:
        raise RuntimeError('Base model context is smaller than the frozen data context.')
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('The fixed protocol requires a CUDA GPU.')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('The fixed protocol requires GPU bfloat16 support.')
    tokenizer=spm.SentencePieceProcessor(model_proto=paths['tokenizer'].read_bytes())
    context=protocol['data']['maximum_encoded_tokens']
    train_pools=load_grouped_pairs(paths['train'],tokenizer,context,protocol,'train')
    validation_pools=load_grouped_pairs(paths['validation'],tokenizer,context,protocol,'validation')
    return protocol,project_root,paths,verified,base,train_pools,validation_pools


def preflight(protocol_path,report_path):
    started=time.perf_counter()
    protocol,project_root,paths,verified,base,train_pools,validation_pools=prepare(protocol_path)
    settings=protocol['preflight']
    rng=random.Random(protocol['training']['seed'])
    observed=Counter(choose_group(rng,protocol['data']['training_sampling_weights'])
                     for _ in range(settings['simulation_draws']))
    observed_ratios={group:observed[group]/settings['simulation_draws'] for group in GROUP_ORDER}
    errors={group:abs(observed_ratios[group]-protocol['data']['training_sampling_weights'][group])
            for group in GROUP_ORDER}
    if max(errors.values())>settings['maximum_sampling_absolute_error']:
        raise RuntimeError(f'Sampling simulation exceeded tolerance: {errors}')
    seed=protocol['training']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device=torch.device('cuda')
    model,targets=create_model(base,protocol['training'],device)
    batch_size=protocol['training']['batch_size']
    longest=sorted((pair for pool in train_pools.values() for pair in pool),
                   key=lambda pair:len(pair[0]),reverse=True)[:batch_size]
    model.train()
    x,y=make_direct_batch(longest,device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x)
        loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
    if not torch.isfinite(loss):
        raise RuntimeError('Preflight produced a non-finite loss.')
    loss.backward()
    gradient_norm=torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        protocol['training']['gradient_clip_norm'],error_if_nonfinite=True)
    peak_memory=torch.cuda.max_memory_allocated(device)
    report={
        'kind':'petlm_final_sft_protocol_preflight_v1',
        'status':'pass_no_training_started',
        'protocol':str(protocol_path.resolve()),
        'protocol_sha256':checksum(protocol_path),
        'verified_input_sha256':verified,
        'encoded_rows':{
            'train':sum(map(len,train_pools.values())),
            'validation':sum(map(len,validation_pools.values()))},
        'encoded_rows_by_group':{
            'train':{group:len(train_pools[group]) for group in GROUP_ORDER},
            'validation':{group:len(validation_pools[group]) for group in GROUP_ORDER}},
        'sampling_simulation':{
            'draws':settings['simulation_draws'],
            'seed':seed,
            'observed_counts':dict(observed),
            'observed_ratios':observed_ratios,
            'absolute_errors':errors,
            'maximum_allowed_error':settings['maximum_sampling_absolute_error']},
        'gpu_check':{
            'name':torch.cuda.get_device_name(device),
            'bfloat16':torch.cuda.is_bf16_supported(),
            'batch_size':batch_size,
            'sequence_length':x.shape[1],
            'loss':loss.item(),
            'gradient_norm_before_clip':float(gradient_norm),
            'peak_memory_bytes':peak_memory,
            'trainable_parameters':trainable_parameter_count(model),
            'lora_targets':targets},
        'formal_checkpoint_created':False,
        'independent_test_read_or_used':False,
        'seconds':time.perf_counter()-started}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    del logits,loss,x,y,model,base
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def train(protocol_path,output,resume=None):
    if output.exists() and resume is None:
        raise FileExistsError(f'{output} exists; use --resume or choose a new directory.')
    protocol,project_root,paths,verified,base,train_pools,validation_pools=prepare(protocol_path)
    protocol_hash=checksum(protocol_path)
    training=protocol['training']
    seed=training['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device=torch.device('cuda')
    model,targets=create_model(base,training,device)
    optimizer=torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training['optimizer']['learning_rate'],weight_decay=training['optimizer']['weight_decay'])
    rng=random.Random(seed)
    start_step=0
    best_score=float('inf')
    stale=0
    baseline=None
    sample_counts=Counter()
    history=[]
    if resume is not None:
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved.get('kind')!=CHECKPOINT_KIND or saved.get('protocol_sha256')!=protocol_hash:
            raise RuntimeError('Resume checkpoint does not match the frozen protocol.')
        if saved.get('input_sha256')!=verified or saved.get('base_checkpoint_sha256')!=verified['base_checkpoint']:
            raise RuntimeError('Resume checkpoint input hashes do not match.')
        load_adapter_state(model,saved['adapter_state'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['sampling_rng'])
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_step=saved['step']
        best_score=saved['best_weighted_validation_loss']
        stale=saved['stale_evaluations']
        baseline=saved['baseline_validation_by_group']
        sample_counts.update(saved['sample_counts'])
        history=list(saved.get('history',[]))
    output.mkdir(parents=True,exist_ok=True)

    def evaluate_groups():
        values={group:evaluate(model,validation_pools[group],training['batch_size'],device,torch.bfloat16)
                for group in GROUP_ORDER}
        return weighted_score(values,protocol['data']['validation_scoring_weights']),values

    if baseline is None:
        baseline_score,baseline=evaluate_groups()
        best_score=baseline_score
    else:
        baseline_score=weighted_score(baseline,protocol['data']['validation_scoring_weights'])

    def save_checkpoint(name,step,current_score,current_groups,eligible):
        state={
            'kind':CHECKPOINT_KIND,
            'adapter_state':adapter_state_dict(model),
            'optimizer':optimizer.state_dict(),
            'config':base['config'],
            'lora':{**training['lora'],'targets_resolved':targets},
            'step':step,
            'protocol_sha256':protocol_hash,
            'input_sha256':verified,
            'base_checkpoint':str(paths['base_checkpoint']),
            'base_checkpoint_sha256':verified['base_checkpoint'],
            'tokenizer_sha256':verified['tokenizer'],
            'weighted_validation_loss':current_score,
            'validation_by_group':current_groups,
            'baseline_validation_by_group':baseline,
            'best_weighted_validation_loss':best_score,
            'promotion_eligible':eligible,
            'stale_evaluations':stale,
            'sample_counts':dict(sample_counts),
            'history':history,
            'sampling_rng':rng.getstate(),
            'python_rng':random.getstate(),
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all()}
        temporary=output/f'{name}.tmp'
        torch.save(state,temporary)
        temporary.replace(output/f'{name}.pt')

    print('Baseline: '+', '.join(f'{group}={baseline[group]:.4f}' for group in GROUP_ORDER)+
          f', weighted={baseline_score:.4f}',flush=True)
    started=time.perf_counter()
    completed=start_step
    weights=protocol['data']['training_sampling_weights']
    minimum_improvement=training['early_stopping']['minimum_weighted_loss_improvement']
    with SummaryWriter(str(output/'runs')) as writer:
        if start_step==0:
            writer.add_scalar('final_sft/validation_weighted',baseline_score,0)
            for group,value in baseline.items():
                writer.add_scalar(f'final_sft/validation_{group}',value,0)
        for step_index in range(start_step,training['maximum_steps']):
            completed=step_index+1
            lr=learning_rate_for_step(training,completed)
            for parameter_group in optimizer.param_groups:
                parameter_group['lr']=lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum=0.0
            for _ in range(training['gradient_accumulation']):
                pairs,groups=sample_batch(train_pools,weights,training['batch_size'],rng)
                sample_counts.update(groups)
                x,y=make_direct_batch(pairs,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(x)
                    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                                         y.reshape(-1),ignore_index=-100)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite final-SFT loss.')
                (loss/training['gradient_accumulation']).backward()
                loss_sum+=loss.item()/training['gradient_accumulation']
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                training['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            writer.add_scalar('final_sft/train_loss',loss_sum,completed)
            writer.add_scalar('final_sft/learning_rate',lr,completed)
            if completed==1 or completed%10==0:
                print(f'step={completed} reply_loss={loss_sum:.4f} lr={lr:.8g}',flush=True)
            if completed%training['evaluation_interval_steps']==0:
                score,values=evaluate_groups()
                guards=group_guard_results(values,baseline,
                    protocol['checkpoint_promotion']['maximum_relative_regression_from_baseline'])
                eligible=all(guards.values()) and score<=best_score-minimum_improvement
                entry={'step':completed,'train_loss':loss_sum,'learning_rate':lr,
                       'weighted_validation_loss':score,'validation_by_group':values,
                       'group_guards':guards,'promotion_eligible':eligible,
                       'sample_counts':dict(sample_counts)}
                history.append(entry)
                writer.add_scalar('final_sft/validation_weighted',score,completed)
                for group,value in values.items():
                    writer.add_scalar(f'final_sft/validation_{group}',value,completed)
                if eligible:
                    best_score=score
                    stale=0
                    save_checkpoint('best',completed,score,values,True)
                else:
                    stale+=1
                save_checkpoint('latest',completed,score,values,eligible)
                print(f'validation weighted={score:.4f}, guards={guards}, best={best_score:.4f}',flush=True)
                if stale>=training['early_stopping']['patience_evaluations']:
                    print('Early stopping: no promotable improvement within fixed patience.',flush=True)
                    break
    report={
        'kind':'petlm_final_sft_training_report_v1',
        'protocol_sha256':protocol_hash,
        'input_sha256':verified,
        'base_checkpoint_sha256':verified['base_checkpoint'],
        'completed_steps':completed,
        'target_steps':training['maximum_steps'],
        'baseline_weighted_validation_loss':baseline_score,
        'baseline_validation_by_group':baseline,
        'best_weighted_validation_loss':best_score,
        'sample_counts':dict(sample_counts),
        'history':history,
        'seconds':time.perf_counter()-started,
        'independent_test_read_or_used':False}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='history'},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--train',action='store_true')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.preflight:
        if args.output or args.resume:
            parser.error('--preflight cannot use --output or --resume.')
        report=args.report or args.protocol.with_name('final-sft-protocol-preflight-r1.json')
        preflight(args.protocol,report)
    else:
        if args.report:
            parser.error('--report is only valid with --preflight.')
        if args.output is None:
            parser.error('--train requires --output.')
        if args.resume is not None and args.resume.parent.resolve()!=args.output.resolve():
            parser.error('--resume must be a checkpoint inside --output.')
        train(args.protocol,args.output,args.resume)


if __name__=='__main__':
    main()
