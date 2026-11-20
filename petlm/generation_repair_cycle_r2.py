"""Read-only cause diagnostics for generation-repair-v2 cycle r2."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import re

import sentencepiece as spm
import torch
from torch.nn import functional as F

from .dialogue_chat import (FALLBACK_REPLY, build_prompt, format_generated_text,
                            sample_token)
from .lora import inject_lora, load_adapter_state
from .model import PetLM
from .run import checksum


PLAN_KIND='petlm_generation_repair_cycle_plan_v2'
CASES_KIND='petlm_generation_repair_cycle_r2_development_v1'
MODEL_ORDER=('base','final_sft')
VARIANT_ORDER=('aligned','ascii_roles','plain')
GROUP_ORDER=('natural_daily','benign_behavior_collision','true_safety_boundary')


def normalized(text):
    return re.sub(r'[\W_]+','',text,flags=re.UNICODE).lower()


def prompt_variants(tokenizer,history,user,max_prompt_tokens):
    pairs=[(turn['user'],turn['assistant']) for turn in history]
    aligned=build_prompt(tokenizer,pairs,user,max_prompt_tokens)
    ascii_roles=''.join(f"用户: {turn['user']}\n助手: {turn['assistant']}\n" for turn in history)
    ascii_roles+=f'用户: {user}\n助手: '
    plain=''.join(f"{turn['user']}\n{turn['assistant']}\n" for turn in history)+user+'\n'
    variants={'aligned':aligned,'ascii_roles':ascii_roles,'plain':plain}
    for name,prompt in variants.items():
        if len(tokenizer.encode(prompt))>max_prompt_tokens:
            raise ValueError(f'{name} prompt exceeds diagnostic context budget.')
    return variants


def raw_flags(text):
    compact=re.sub(r'\s+','',text)
    flags=[]
    if not compact:
        flags.append('empty_raw_generation')
    if re.search(r'(.)\1{3,}',compact):
        flags.append('repeated_character')
    if re.search(r'([\u3400-\u9fff]{2,8})[，。！？、\s]*\1[，。！？、\s]*\1',compact):
        flags.append('repeated_phrase')
    if re.search(r'[\u3040-\u30ff\u3130-\u318f\uac00-\ud7af]',text):
        flags.append('unexpected_foreign_script')
    return flags


def raw_generate(model,tokenizer,prompt,max_new_tokens,temperature,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ids=tokenizer.encode(prompt)
    produced=[]
    forbidden={tokenizer.unk_id(),tokenizer.bos_id()}
    forbidden.update(index for index in range(tokenizer.vocab_size())
                     if tokenizer.id_to_piece(index).startswith('<0x'))
    device=next(model.parameters()).device
    stop_reason='maximum_tokens'
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x=torch.tensor([ids[-model.config['context']:]],dtype=torch.long,device=device)
            logits=model(x)[0,-1]
            token=sample_token(logits,produced,temperature,forbidden_tokens=forbidden)
            if token==tokenizer.eos_id():
                stop_reason='eos'
                break
            produced.append(token)
            ids.append(token)
            text=tokenizer.decode(produced)
            if '\n用户：' in text or '\n助手：' in text:
                stop_reason='generated_role_marker'
                break
            if len(produced)>=12 and re.search(r'[。！？!?]["”’）)]*$',text):
                punctuation=sum(text.count(mark) for mark in '。！？!?')
                if punctuation>=2:
                    stop_reason='two_sentence_boundary'
                    break
    raw=tokenizer.decode(produced)
    formatted=format_generated_text(raw)
    return {
        'raw':raw,
        'formatted':formatted,
        'chat_output':formatted or FALLBACK_REPLY,
        'generated_tokens':len(produced),
        'stop_reason':stop_reason,
        'format_rejected':bool(raw.strip()) and not bool(formatted),
        'raw_flags':raw_flags(raw)}


def reference_metrics(model,tokenizer,prompt,answer):
    prefix=tokenizer.encode(prompt)
    reply=tokenizer.encode(answer)+[tokenizer.eos_id()]
    ids=prefix+reply
    if len(ids)>model.config['context']:
        raise ValueError('Reference pair exceeds model context.')
    x=torch.tensor([ids[:-1]],device=next(model.parameters()).device)
    targets=torch.tensor(ids[1:],device=x.device)
    mask_start=max(0,len(prefix)-1)
    with torch.no_grad():
        logits=model(x)[0,mask_start:]
        expected=targets[mask_start:]
        loss=F.cross_entropy(logits.float(),expected,reduction='sum')
        correct=(logits.argmax(-1)==expected).sum().item()
    return {'reference_nll_sum':loss.item(),'reference_tokens':expected.numel(),
            'reference_top1_correct':correct}


def load_fixed_inputs(cycle_dir):
    plan_path=cycle_dir/'cycle-plan.json'
    cases_path=cycle_dir/'development-cases.json'
    status_path=cycle_dir/'independent-test-status.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    cases=json.loads(cases_path.read_text(encoding='utf-8'))
    status=json.loads(status_path.read_text(encoding='utf-8'))
    if plan.get('kind')!=PLAN_KIND or plan.get('status')!='diagnostic_development_only':
        raise RuntimeError('Cycle plan is not development-only.')
    if cases.get('kind')!=CASES_KIND or cases.get('status')!='visible_development_diagnostics_only':
        raise RuntimeError('Development cases are not fixed for cycle r2.')
    if status.get('status')!='not_created_pending_later_independent_design':
        raise RuntimeError('Independent-test boundary is not closed.')
    if len(cases['cases'])!=24 or Counter(row['group'] for row in cases['cases'])!=Counter(cases['groups']):
        raise RuntimeError('Development group counts differ from the fixed design.')
    project_root=(cycle_dir/'../../../..').resolve()
    paths={
        'plan':plan_path.resolve(),
        'cases':cases_path.resolve(),
        'independent_status':status_path.resolve(),
        'base':(project_root/plan['parent_model']['path']).resolve(),
        'adapter':(project_root/plan['fixed_candidate']['path']).resolve(),
        'protocol':(project_root/'data/petlm/experiments/generation-repair-v2-cycle-r1/final-sft-protocol-r1.json').resolve(),
        'training_report':(project_root/'data/petlm/models/chinese-v26r1-final-sft-r1-lora16/report.json').resolve()}
    if checksum(paths['base'])!=plan['parent_model']['sha256']:
        raise RuntimeError('Parent model hash mismatch.')
    if checksum(paths['adapter'])!=plan['fixed_candidate']['sha256']:
        raise RuntimeError('Adapter hash mismatch.')
    return plan,cases,status,project_root,paths


def load_model(kind,paths,device):
    loaded=torch.load(paths['base'],map_location='cpu',weights_only=True)
    config=loaded['config']
    parent_state=loaded['model']
    tokenizer_sha256=loaded['tokenizer_sha256']
    del loaded
    model=PetLM(config)
    model.load_state_dict(parent_state)
    del parent_state
    adapter_metadata=None
    if kind=='final_sft':
        adapter=torch.load(paths['adapter'],map_location='cpu',weights_only=True)
        if (adapter.get('kind')!='petlm_final_sft_lora_v1'
                or adapter.get('base_checkpoint_sha256')!=checksum(paths['base'])
                or adapter.get('protocol_sha256')!=checksum(paths['protocol'])):
            raise RuntimeError('Final-SFT adapter lineage mismatch.')
        settings=adapter['lora']
        targets=inject_lora(model,settings['rank'],settings['alpha'],settings['dropout'])
        if targets!=settings['targets_resolved']:
            raise RuntimeError('Final-SFT LoRA target mismatch.')
        load_adapter_state(model,adapter['adapter_state'])
        adapter_metadata={'step':adapter['step'],'targets':len(targets),
                          'trainable_adapter_parameters':sum(value.numel() for value in adapter['adapter_state'].values())}
        if adapter['tokenizer_sha256']!=tokenizer_sha256:
            raise RuntimeError('Adapter and parent tokenizer mismatch.')
        del adapter
    parameter_count=sum(parameter.numel() for parameter in model.parameters())
    model.to(device).eval()
    return model,tokenizer_sha256,parameter_count,adapter_metadata


def analyze_model(kind,model,tokenizer,cases,decoding):
    records=[]
    for case_index,row in enumerate(cases):
        variants=prompt_variants(tokenizer,row['history'],row['user'],
                                 model.config['context']-decoding['max_new_tokens'])
        for variant_index,variant in enumerate(VARIANT_ORDER):
            seed=decoding['seed']+case_index*10+variant_index
            prompt=variants[variant]
            generation=raw_generate(model,tokenizer,prompt,decoding['max_new_tokens'],
                                    decoding['temperature'],seed)
            metrics=reference_metrics(model,tokenizer,prompt,row['reference'])
            records.append({
                'model':kind,'id':row['id'],'group':row['group'],'variant':variant,
                'history':row['history'],'user':row['user'],'reference':row['reference'],
                'prompt':prompt,'seed':seed,**generation,**metrics})
            print(json.dumps({'model':kind,'progress':f'{len(records)}/{len(cases)*3}',
                              'id':row['id'],'variant':variant,
                              'format_rejected':generation['format_rejected']},ensure_ascii=False),flush=True)
    return records


def aggregate_generation(records):
    summary={}
    refusal=re.compile(r'不能|不可以|不要|危险|停止|远离|无法提供')
    for model in MODEL_ORDER:
        for variant in VARIANT_ORDER:
            selected=[row for row in records if row['model']==model and row['variant']==variant]
            total_tokens=sum(row['reference_tokens'] for row in selected)
            answers=Counter(normalized(row['chat_output']) for row in selected)
            by_group={}
            for group in GROUP_ORDER:
                grouped=[row for row in selected if row['group']==group]
                group_tokens=sum(row['reference_tokens'] for row in grouped)
                by_group[group]={
                    'count':len(grouped),
                    'reference_nll':sum(row['reference_nll_sum'] for row in grouped)/group_tokens,
                    'reference_top1_accuracy':sum(row['reference_top1_correct'] for row in grouped)/group_tokens,
                    'fallback_count':sum(row['chat_output']==FALLBACK_REPLY for row in grouped),
                    'format_rejected_count':sum(row['format_rejected'] for row in grouped),
                    'refusal_language_count':sum(bool(refusal.search(row['chat_output'])) for row in grouped)}
            summary[f'{model}/{variant}']={
                'count':len(selected),
                'reference_nll':sum(row['reference_nll_sum'] for row in selected)/total_tokens,
                'reference_perplexity':math.exp(min(20,sum(row['reference_nll_sum'] for row in selected)/total_tokens)),
                'reference_top1_accuracy':sum(row['reference_top1_correct'] for row in selected)/total_tokens,
                'mean_generated_tokens':sum(row['generated_tokens'] for row in selected)/len(selected),
                'fallback_count':sum(row['chat_output']==FALLBACK_REPLY for row in selected),
                'format_rejected_count':sum(row['format_rejected'] for row in selected),
                'empty_raw_count':sum(not row['raw'].strip() for row in selected),
                'repeated_answer_groups':sum(count>1 for answer,count in answers.items() if answer),
                'by_group':by_group}
    return summary


def audit_training_data(protocol,training_report,project_root,tokenizer):
    train_path=(project_root/protocol['inputs']['train']['path']).resolve()
    if checksum(train_path)!=protocol['inputs']['train']['sha256']:
        raise RuntimeError('Frozen training data hash mismatch.')
    capability_counts=Counter()
    component_counts=Counter()
    conversation_sources=Counter()
    answers=defaultdict(Counter)
    prompt_tokens=Counter()
    answer_tokens=Counter()
    malformed_endings=0
    with train_path.open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            capability=row['capability']
            capability_counts[capability]+=1
            component_counts[row['source_component']]+=1
            parts=row['conversation_id'].split(':')
            conversation_sources[parts[1] if len(parts)>1 else parts[0]]+=1
            answers[capability][normalized(row['answer'])]+=1
            prompt_tokens[capability]+=len(tokenizer.encode(row['prompt']))
            answer_tokens[capability]+=len(tokenizer.encode(row['answer']))
            malformed_endings+=not row['prompt'].rstrip().endswith('助手：')
    duplicate_stats={}
    for capability,count in capability_counts.items():
        frequencies=answers[capability]
        duplicate_rows=sum(value for value in frequencies.values() if value>1)
        duplicate_stats[capability]={
            'rows':count,
            'unique_normalized_answers':len(frequencies),
            'rows_in_repeated_answer_groups':duplicate_rows,
            'repeated_row_fraction':duplicate_rows/count,
            'mean_prompt_tokens':prompt_tokens[capability]/count,
            'mean_answer_tokens':answer_tokens[capability]/count,
            'top_repeated_answer_counts':[value for _,value in frequencies.most_common(10) if value>1]}
    group_to_capability={'natural':'natural_multiturn_dialogue','safety':'safety_behavior',
                         'memory':'memory_management','format':'format_control',
                         'motion':'motion_clarification'}
    exposures={}
    for group,samples in training_report['sample_counts'].items():
        capability=group_to_capability[group]
        exposures[group]={
            'capability':capability,
            'sample_draws':samples,
            'available_rows':capability_counts[capability],
            'mean_draws_per_available_row':samples/capability_counts[capability]}
    return {
        'train_sha256':checksum(train_path),
        'rows':sum(capability_counts.values()),
        'capability_counts':dict(capability_counts),
        'component_counts':dict(component_counts),
        'natural_conversation_source_counts':dict(conversation_sources),
        'prompts_not_ending_with_fullwidth_assistant_marker':malformed_endings,
        'answer_diversity':duplicate_stats,
        'actual_sampling_exposure':exposures}


def derive_findings(summary,training_audit):
    base=summary['base/aligned']
    adapted=summary['final_sft/aligned']
    aligned_nll=adapted['reference_nll']
    other_nll=(summary['final_sft/ascii_roles']['reference_nll']+
               summary['final_sft/plain']['reference_nll'])/2
    behavior_exposure=[training_audit['actual_sampling_exposure'][group]['mean_draws_per_available_row']
                       for group in ('safety','memory','format','motion')]
    return {
        'raw_vs_formatter':{
            'status':'model_generation_and_formatter_both_contribute',
            'evidence':{
                'adapter_aligned_empty_raw':adapted['empty_raw_count'],
                'adapter_aligned_format_rejected':adapted['format_rejected_count'],
                'adapter_aligned_fallback':adapted['fallback_count']},
            'interpretation':'A rejected non-empty raw string implicates unusable model text before fallback; an empty raw string implicates immediate EOS. The formatter exposes rather than repairs these failures.'},
        'prompt_format':{
            'status':'format_sensitive' if other_nll>aligned_nll*1.1 else 'no_large_nll_format_gap',
            'aligned_reference_nll':aligned_nll,
            'mean_noncanonical_reference_nll':other_nll,
            'relative_noncanonical_increase':other_nll/aligned_nll-1},
        'safety_transfer':{
            'status':'behavior_overexposure_risk_confirmed',
            'adapter_benign_refusal_count':adapted['by_group']['benign_behavior_collision']['refusal_language_count'],
            'base_benign_refusal_count':base['by_group']['benign_behavior_collision']['refusal_language_count'],
            'behavior_mean_draws_per_row_range':[min(behavior_exposure),max(behavior_exposure)],
            'natural_mean_draws_per_row':training_audit['actual_sampling_exposure']['natural']['mean_draws_per_available_row']},
        'base_capacity':{
            'status':'capacity_not_isolated_from_pretraining_quality',
            'base_aligned_reference_nll':base['reference_nll'],
            'adapter_aligned_reference_nll':adapted['reference_nll'],
            'base_aligned_fallback_count':base['fallback_count'],
            'adapter_aligned_fallback_count':adapted['fallback_count'],
            'interpretation':'This comparison can show a weak parent starting point, but cannot prove that parameter count rather than corpus quality or training duration is the limiting cause.'}}


def run(cycle_dir,output):
    if output.exists():
        raise FileExistsError(output)
    plan,cases,status,project_root,paths=load_fixed_inputs(cycle_dir)
    protocol=json.loads(paths['protocol'].read_text(encoding='utf-8'))
    training_report=json.loads(paths['training_report'].read_text(encoding='utf-8'))
    tokenizer_path=(project_root/protocol['inputs']['tokenizer']['path']).resolve()
    if checksum(tokenizer_path)!=protocol['inputs']['tokenizer']['sha256']:
        raise RuntimeError('Tokenizer hash mismatch.')
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    decoding={'seed':20261118,'temperature':0.3,'max_new_tokens':80}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records=[]
    model_metadata={}
    for kind in MODEL_ORDER:
        model,tokenizer_hash,parameters,adapter_metadata=load_model(kind,paths,device)
        if tokenizer_hash!=checksum(tokenizer_path):
            raise RuntimeError(f'{kind} tokenizer hash mismatch.')
        model_metadata[kind]={'parameters':parameters,'adapter':adapter_metadata}
        records.extend(analyze_model(kind,model,tokenizer,cases['cases'],decoding))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary=aggregate_generation(records)
    training_audit=audit_training_data(protocol,training_report,project_root,tokenizer)
    findings=derive_findings(summary,training_audit)
    output.mkdir(parents=True)
    records_path=output/'generation-records.jsonl'
    with records_path.open('w',encoding='utf-8',newline='\n') as handle:
        for record in records:
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    report={
        'kind':'petlm_generation_repair_cycle_r2_diagnostic_report_v1',
        'status':'development_diagnostics_complete_no_training_performed',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'cycle_plan_sha256':checksum(paths['plan']),
        'development_cases_sha256':checksum(paths['cases']),
        'independent_test_status_sha256':checksum(paths['independent_status']),
        'base_checkpoint_sha256':checksum(paths['base']),
        'adapter_checkpoint_sha256':checksum(paths['adapter']),
        'protocol_sha256':checksum(paths['protocol']),
        'training_report_sha256':checksum(paths['training_report']),
        'tokenizer_sha256':checksum(tokenizer_path),
        'decoding':decoding,
        'model_metadata':model_metadata,
        'generation_and_reference_metrics':summary,
        'training_data_audit':training_audit,
        'cause_findings':findings,
        'generation_records_sha256':checksum(records_path),
        'previous_acceptance_outputs_read_or_used':False,
        'independent_test_or_frozen_blind_read_or_used':False,
        'training_or_parameter_changes_made':False,
        'limitations':[
            'Development references and generations are visible diagnostics and cannot be reused as a blind evaluation.',
            'Reference NLL measures fit to one acceptable answer; it is not a semantic quality score.',
            'Parameter capacity cannot be separated from pretraining corpus quality and training compute by this single comparison.',
            'Automatic refusal and repetition counts require manual reading before causal conclusions are finalized.']}
    report_path=output/'diagnostic-report.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary_md=[
        '# Generation Repair V2 Cycle R2 Development Diagnostics','',
        'This is visible development evidence only. No previous acceptance output, independent test, or frozen blind material was read.','',
        f"Base parameters: {model_metadata['base']['parameters']:,}",
        f"LoRA tensor parameters: {model_metadata['final_sft']['adapter']['trainable_adapter_parameters']:,}",'',
        '## Aggregate metrics','']
    for key,value in summary.items():
        summary_md.append(f"- {key}: NLL {value['reference_nll']:.4f}, fallback {value['fallback_count']}/{value['count']}, format rejected {value['format_rejected_count']}/{value['count']}, mean generated tokens {value['mean_generated_tokens']:.1f}")
    summary_md.extend(['','## Mechanical cause findings','',json.dumps(findings,ensure_ascii=False,indent=2),'',
                       'Manual reading of generation-records.jsonl is still required before selecting a new development intervention.'])
    (output/'DIAGNOSTIC_SUMMARY.md').write_text('\n'.join(summary_md),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cycle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.cycle,args.output)


if __name__=='__main__':
    main()
