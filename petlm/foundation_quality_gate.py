"""Evaluate raw foundation Chinese continuations against a fixed visible gate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import re

import sentencepiece as spm
import torch
from torch.nn import functional as F

from .dialogue_chat import sample_token
from .model import PetLM
from .run import checksum


PROTOCOL_KIND='petlm_foundation_chinese_quality_gate_v1'
CASES_KIND='petlm_foundation_chinese_quality_development_v1'


def normalized(text):
    return re.sub(r'[\W_]+','',text,flags=re.UNICODE).lower()


def mechanical_flags(text,stop_reason):
    compact=re.sub(r'\s+','',text)
    flags=[]
    chinese=re.findall(r'[\u3400-\u9fff]',compact)
    if not compact:
        flags.append('empty_generation')
    if len(chinese)<10:
        flags.append('too_few_chinese_characters')
    if compact and len(chinese)/len(compact)<0.65:
        flags.append('low_chinese_ratio')
    if re.search(r'[\u3040-\u30ff\u3130-\u318f\uac00-\ud7af]',compact):
        flags.append('unexpected_foreign_script')
    if re.search(r'(.)\1{3,}',compact):
        flags.append('repeated_character')
    if re.search(r'([\u3400-\u9fff]{2,10})[，。！？、\s]*\1[，。！？、\s]*\1',compact):
        flags.append('repeated_phrase')
    grams=[compact[index:index+4] for index in range(max(0,len(compact)-3))]
    if any(grams.count(gram)>=3 for gram in set(grams) if len(gram)==4):
        flags.append('repeated_four_character_loop')
    if re.search(r'(?:用户|助手)\s*[:：]',text):
        flags.append('unexpected_dialogue_role_marker')
    if compact and not re.search(r'[。！？!?]["”’）)]*$',compact):
        flags.append('incomplete_sentence_ending')
    if stop_reason=='maximum_tokens':
        flags.append('maximum_token_truncation')
    return flags


def generate(model,tokenizer,prompt,decoding,seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ids=tokenizer.encode(prompt)
    if not ids or len(ids)+decoding['max_new_tokens']>model.config['context']:
        raise ValueError('Development prompt does not fit the foundation context.')
    produced=[]
    forbidden={tokenizer.unk_id(),tokenizer.bos_id()}
    forbidden.update(index for index in range(tokenizer.vocab_size())
                     if tokenizer.id_to_piece(index).startswith('<0x'))
    device=next(model.parameters()).device
    stop_reason='maximum_tokens'
    with torch.no_grad():
        for _ in range(decoding['max_new_tokens']):
            x=torch.tensor([ids],dtype=torch.long,device=device)
            logits=model(x)[0,-1]
            token=sample_token(logits,produced,decoding['temperature'],
                               decoding['repetition_penalty'],decoding['top_k'],forbidden)
            if token==tokenizer.eos_id():
                stop_reason='eos'
                break
            produced.append(token)
            ids.append(token)
            text=tokenizer.decode(produced)
            sentences=sum(text.count(mark) for mark in '。！？!?')
            if (sentences>=decoding['maximum_complete_sentences']
                    and re.search(r'[。！？!?]["”’）)]*$',text)):
                stop_reason='sentence_limit'
                break
    text=tokenizer.decode(produced).strip()
    return text,len(produced),stop_reason,mechanical_flags(text,stop_reason)


def reference_metrics(model,tokenizer,prompt,reference):
    prefix=tokenizer.encode(prompt)
    continuation=tokenizer.encode(reference)+[tokenizer.eos_id()]
    ids=prefix+continuation
    x=torch.tensor([ids[:-1]],device=next(model.parameters()).device)
    expected=torch.tensor(ids[1:],device=x.device)
    start=len(prefix)-1
    with torch.no_grad():
        logits=model(x)[0,start:]
        targets=expected[start:]
        loss=F.cross_entropy(logits.float(),targets,reduction='sum')
        correct=(logits.argmax(-1)==targets).sum().item()
    return loss.item(),targets.numel(),correct


def load_protocol(protocol_path,checkpoint=None):
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('kind')!=PROTOCOL_KIND or protocol.get('status')!='frozen_before_new_foundation_training':
        raise RuntimeError('Foundation gate protocol is not frozen.')
    project_root=(protocol_path.parent/'../../../..').resolve()
    cases_path=(project_root/protocol['development_cases']['path']).resolve()
    tokenizer_path=(project_root/protocol['tokenizer']['path']).resolve()
    checkpoint_path=(checkpoint.resolve() if checkpoint else
                     (project_root/protocol['baseline_checkpoint']['path']).resolve())
    if checksum(cases_path)!=protocol['development_cases']['sha256']:
        raise RuntimeError('Foundation development cases changed.')
    if checksum(tokenizer_path)!=protocol['tokenizer']['sha256']:
        raise RuntimeError('Foundation tokenizer changed.')
    if checkpoint is None and checksum(checkpoint_path)!=protocol['baseline_checkpoint']['sha256']:
        raise RuntimeError('Foundation baseline checkpoint changed.')
    cases=json.loads(cases_path.read_text(encoding='utf-8'))
    if cases.get('kind')!=CASES_KIND or len(cases['cases'])!=protocol['development_cases']['count']:
        raise RuntimeError('Foundation development case structure mismatch.')
    return protocol,cases,checkpoint_path,tokenizer_path


def evaluate(protocol_path,output,checkpoint=None,label='baseline'):
    if output.exists():
        raise FileExistsError(output)
    protocol,cases,checkpoint_path,tokenizer_path=load_protocol(protocol_path,checkpoint)
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if state.get('tokenizer_sha256')!=checksum(tokenizer_path):
        raise RuntimeError('Checkpoint and gate tokenizer differ.')
    model=PetLM(state['config'])
    model.load_state_dict(state['model'])
    del state
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    records=[]
    totals={'loss':0.0,'tokens':0,'correct':0}
    decoding=protocol['decoding']
    for index,case in enumerate(cases['cases']):
        seed=decoding['seed']+index
        continuation,generated_tokens,stop_reason,flags=generate(
            model,tokenizer,case['prompt'],decoding,seed)
        loss,tokens,correct=reference_metrics(model,tokenizer,case['prompt'],case['reference'])
        totals['loss']+=loss
        totals['tokens']+=tokens
        totals['correct']+=correct
        record={
            'id':case['id'],'category':case['category'],'prompt':case['prompt'],
            'reference':case['reference'],'continuation':continuation,
            'full_text':case['prompt']+continuation,'seed':seed,
            'generated_tokens':generated_tokens,'stop_reason':stop_reason,
            'mechanical_flags':flags,'mechanical_pass':not flags,
            'reference_nll_sum':loss,'reference_tokens':tokens,
            'reference_top1_correct':correct,
            'human_scores':{
                'relevance_to_prefix':None,'local_coherence':None,
                'grammar_and_sentence_completeness':None,'natural_chinese':None,
                'non_repetition':None,'critical_failure':None,'notes':''}}
        records.append(record)
        print(json.dumps({'progress':f'{index+1}/{len(cases["cases"])}','id':case['id'],
                          'flags':flags},ensure_ascii=False),flush=True)
    output.mkdir(parents=True)
    records_path=output/'outputs.jsonl'
    with records_path.open('w',encoding='utf-8',newline='\n') as handle:
        for record in records:
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    frequencies=Counter(normalized(row['continuation']) for row in records)
    repeated={text:count for text,count in frequencies.items() if text and count>1}
    nll=totals['loss']/totals['tokens']
    mechanical_passes=sum(row['mechanical_pass'] for row in records)
    metrics={
        'reference_nll':nll,'reference_perplexity':math.exp(min(20,nll)),
        'reference_top1_accuracy':totals['correct']/totals['tokens'],
        'mechanical_passes':mechanical_passes,
        'mechanical_pass_fraction':mechanical_passes/len(records),
        'exact_repeated_completion_groups':repeated,
        'mean_generated_tokens':sum(row['generated_tokens'] for row in records)/len(records),
        'flag_counts':dict(Counter(flag for row in records for flag in row['mechanical_flags']))}
    thresholds=protocol['development_pass_thresholds']
    automatic_checks={
        'reference_nll_absolute':nll<=thresholds['reference_nll_maximum'],
        'reference_top1_accuracy':metrics['reference_top1_accuracy']>=thresholds['reference_top1_accuracy_minimum'],
        'mechanical_pass_fraction':metrics['mechanical_pass_fraction']>=thresholds['mechanical_pass_fraction_minimum'],
        'exact_repeated_completion_groups':len(repeated)<=thresholds['exact_repeated_completion_groups_maximum']}
    report={
        'kind':'petlm_foundation_chinese_quality_gate_evaluation_v1',
        'status':'baseline_recorded_pending_human_review' if label=='baseline' else 'candidate_recorded_pending_human_review',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'label':label,'checkpoint':str(checkpoint_path),
        'checkpoint_sha256':checksum(checkpoint_path),
        'protocol_sha256':checksum(protocol_path),
        'cases_sha256':checksum((protocol_path.parent/'development-cases.json').resolve()),
        'tokenizer_sha256':checksum(tokenizer_path),'device':str(device),
        'decoding':decoding,'metrics':metrics,'automatic_checks':automatic_checks,
        'automatic_gate_pass':all(automatic_checks.values()),
        'human_review_complete':False,'human_gate_pass':None,
        'training_or_parameter_changes_made':False,
        'old_dialogue_acceptance_or_blind_data_read':False,
        'outputs_sha256':checksum(records_path),
        'limitations':[
            'Mechanical checks cannot establish semantic coherence or factual correctness.',
            'Reference NLL scores one valid continuation and is not a complete language-quality metric.',
            'Visible development outputs cannot serve as the later independent foundation gate.']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    review=['# 基础模型中文质量开发审核','',
            '这是可见开发集，不是独立测试。逐条检查续写是否承接前文、语法完整、自然且没有无意义重复。','']
    for row in records:
        review.extend([
            f"## {row['id']} | {row['category']}",'',
            f"前文：{row['prompt']}",'',f"模型续写：{row['continuation']}",'',
            f"完整文本：{row['full_text']}",'',
            '自动标记：'+('、'.join(row['mechanical_flags']) if row['mechanical_flags'] else '无'),'',
            '评分：承接相关 __/2；局部连贯 __/2；语法完整 __/2；自然中文 __/2；无重复 __/2','',
            '关键失败：否 / 是（说明：________________）','',
            '审核备注：________________',''])
    review.extend(['## 汇总','','通过条数：__/24','','关键失败数：__','',
                   '开发门槛：通过 / 不通过','','审核人及日期：________________',''])
    (output/'HUMAN_REVIEW.md').write_text('\n'.join(review),encoding='utf-8')
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--label',default='baseline')
    args=parser.parse_args()
    evaluate(args.protocol,args.output,args.checkpoint,args.label)


if __name__=='__main__':
    main()
