"""Generate a visible, pure-model human acceptance packet for final SFT."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import sentencepiece as spm
import torch

from .dialogue_chat import FALLBACK_REPLY, reply
from .lora import inject_lora, load_adapter_state
from .model import PetLM
from .run import checksum


CHECKPOINT_KIND='petlm_final_sft_lora_v1'
CASES_KIND='petlm_final_sft_visible_human_acceptance_cases_v1'


def normalized(text):
    return re.sub(r'[\W_]+','',text,flags=re.UNICODE).lower()


def diagnostic_flags(answer,previous_answers):
    flags=[]
    compact=re.sub(r'\s+','',answer)
    if not compact:
        flags.append('empty_response')
    if FALLBACK_REPLY in answer:
        flags.append('fallback_response')
    if re.search(r'[\u3040-\u30ff\u3130-\u318f\uac00-\ud7af]',answer):
        flags.append('unexpected_foreign_script')
    if re.search(r'(.)\1{3,}',compact):
        flags.append('repeated_character')
    if re.search(r'([\u3400-\u9fff]{2,8})[，。！？、\s]*\1[，。！？、\s]*\1',compact):
        flags.append('repeated_phrase')
    if normalized(answer) and normalized(answer) in {normalized(item) for item in previous_answers}:
        flags.append('exact_answer_repeated_in_conversation')
    if any(term in answer for term in ('论文检索','检索论文','不能核实确切出处')):
        flags.append('unrelated_research_redirect')
    if len(compact)>=15 and not re.search(r'[。！？!?；;：:]$',compact):
        flags.append('possibly_incomplete_ending')
    return flags


def load_model(cases_path):
    cases=json.loads(cases_path.read_text(encoding='utf-8'))
    if cases.get('kind')!=CASES_KIND or cases.get('status')!='fixed_after_training_not_blind':
        raise RuntimeError('Acceptance cases are not fixed visible post-training cases.')
    project_root=(cases_path.parent/'../../../..').resolve()
    checkpoint_path=(project_root/cases['checkpoint']['path']).resolve()
    if checksum(checkpoint_path)!=cases['checkpoint']['sha256']:
        raise RuntimeError('Best checkpoint hash differs from the fixed acceptance cases.')
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if state.get('kind')!=CHECKPOINT_KIND or state.get('promotion_eligible') is not True:
        raise RuntimeError('Checkpoint is not a promotable final-SFT LoRA checkpoint.')
    base_path=Path(state['base_checkpoint'])
    if checksum(base_path)!=state['base_checkpoint_sha256']:
        raise RuntimeError('Base checkpoint hash does not match the adapter.')
    protocol_path=project_root/'data/petlm/experiments/generation-repair-v2-cycle-r1/final-sft-protocol-r1.json'
    if checksum(protocol_path)!=state['protocol_sha256']:
        raise RuntimeError('Frozen training protocol hash does not match the adapter.')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    tokenizer_path=(project_root/protocol['inputs']['tokenizer']['path']).resolve()
    if checksum(tokenizer_path)!=state['tokenizer_sha256']:
        raise RuntimeError('Tokenizer hash does not match the adapter.')
    loaded=torch.load(base_path,map_location='cpu',weights_only=True)
    base_config=loaded['config']
    parent_state=loaded['model']
    del loaded
    model=PetLM(base_config)
    model.load_state_dict(parent_state)
    del parent_state
    lora=state['lora']
    targets=inject_lora(model,lora['rank'],lora['alpha'],lora['dropout'])
    if targets!=lora['targets_resolved']:
        raise RuntimeError('Resolved LoRA targets do not match the checkpoint.')
    load_adapter_state(model,state['adapter_state'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    metadata={
        'project_root':project_root,
        'checkpoint_path':checkpoint_path,
        'checkpoint_sha256':cases['checkpoint']['sha256'],
        'base_checkpoint_sha256':state['base_checkpoint_sha256'],
        'protocol_sha256':state['protocol_sha256'],
        'tokenizer_sha256':state['tokenizer_sha256'],
        'checkpoint_step':state['step'],
        'device':str(device)}
    return cases,model,tokenizer,metadata


def generate(cases_path,output):
    if output.exists():
        raise FileExistsError(output)
    cases,model,tokenizer,metadata=load_model(cases_path)
    decoding=cases['decoding']
    output.mkdir(parents=True)
    records=[]
    all_previous=[]
    for conversation_index,conversation in enumerate(cases['conversations']):
        history=[]
        thread_answers=[]
        for turn_index,turn in enumerate(conversation['turns']):
            seed=decoding['seed']+conversation_index*100+turn_index
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            answer=reply(model,tokenizer,history,turn['user'],
                         max_new_tokens=decoding['max_new_tokens'],
                         temperature=decoding['temperature'],use_protection=False)
            flags=diagnostic_flags(answer,thread_answers)
            record={
                'id':f"{conversation['id']}-t{turn_index+1}",
                'conversation_id':conversation['id'],
                'turn':turn_index+1,
                'focus':conversation['focus'],
                'history':[{'user':user,'assistant':assistant} for user,assistant in history],
                'user':turn['user'],
                'answer':answer,
                'review_checks':turn['checks'],
                'seed':seed,
                'automatic_flags':flags,
                'human_scores':{
                    'relevance':None,
                    'context_coherence':None,
                    'grammar_completeness':None,
                    'naturalness':None,
                    'non_repetition':None,
                    'critical_failure':None,
                    'notes':''}}
            records.append(record)
            history.append((turn['user'],answer))
            thread_answers.append(answer)
            all_previous.append(answer)
            print(json.dumps({'progress':f'{len(records)}/30','id':record['id'],
                              'flags':flags},ensure_ascii=False),flush=True)
    output_jsonl=output/'outputs.jsonl'
    with output_jsonl.open('w',encoding='utf-8',newline='\n') as handle:
        for record in records:
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    exact=Counter(normalized(record['answer']) for record in records)
    repeated={answer:count for answer,count in exact.items() if answer and count>1}
    automatic_counts=Counter(flag for record in records for flag in record['automatic_flags'])
    report={
        'kind':'petlm_final_sft_visible_human_acceptance_generation_v1',
        'status':'generated_pending_human_review',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'cases':str(cases_path.resolve()),
        'cases_sha256':checksum(cases_path),
        'checkpoint':str(metadata['checkpoint_path']),
        'checkpoint_sha256':metadata['checkpoint_sha256'],
        'base_checkpoint_sha256':metadata['base_checkpoint_sha256'],
        'protocol_sha256':metadata['protocol_sha256'],
        'tokenizer_sha256':metadata['tokenizer_sha256'],
        'checkpoint_step':metadata['checkpoint_step'],
        'device':metadata['device'],
        'conversation_count':len(cases['conversations']),
        'turn_count':len(records),
        'decoding':decoding,
        'pure_model_generation':True,
        'protection_or_state_machine_used':False,
        'automatic_flag_counts':dict(automatic_counts),
        'exact_repeated_answer_groups':repeated,
        'outputs_sha256':checksum(output_jsonl),
        'human_review_complete':False,
        'human_acceptance_result':None,
        'old_frozen_or_independent_sets_read':False,
        'training_or_parameter_changes_made':False,
        'limitations':[
            'Automatic flags are screening hints and are not semantic or grammatical judgments.',
            'These visible cases are an internal human acceptance packet, not an unseen blind set.',
            'Results may decide whether to proceed, but these cases may not be used for further training or parameter tuning.']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    review=[
        '# Final SFT R1 日常多轮对话人工验收',
        '',
        '状态：待逐条人工审核。这是训练后可见验收，不是冻结盲测。',
        '',
        '每轮按相关性、上下文连贯、语法完整、表达自然、无无意义重复分别评 0-2 分。总分至少 8 分为该轮通过；任一严重答非所问、语义不明、无意义重复、格式违约或危险建议均为关键失败。',
        '']
    for record in records:
        review.extend([
            f"## {record['id']} | {record['focus']}",
            '',
            f"用户：{record['user']}",
            '',
            f"模型：{record['answer']}",
            '',
            '审核重点：'+'；'.join(record['review_checks']),
            '',
            '自动标记：'+('、'.join(record['automatic_flags']) if record['automatic_flags'] else '无'),
            '',
            '评分：相关性 __/2；上下文连贯 __/2；语法完整 __/2；表达自然 __/2；无无意义重复 __/2',
            '',
            '关键失败：否 / 是（说明：________________）',
            '',
            '审核备注：________________',
            ''])
    review.extend([
        '## 汇总', '',
        '完成审核：否', '',
        '通过轮数：__/30', '',
        '关键失败数：__', '',
        '最终结论：通过 / 不通过', '',
        '审核人及日期：________________', ''])
    (output/'HUMAN_REVIEW.md').write_text('\n'.join(review),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    generate(args.cases,args.output)


if __name__=='__main__':
    main()
