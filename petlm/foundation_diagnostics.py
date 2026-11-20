"""Read-only model/data diagnostics; all new outputs go to a new cycle directory.

Never trains, reads old blind cases, overwrites results, or assigns human scores.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import sentencepiece as spm
import torch

from .model import PetLM
from .run import checksum


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def compact(text):
    return re.sub(r'\s+', '', text)


def category(text):
    if re.search('厨房|烹饪|食材|散步|家务|打扫|早餐', text):
        return 'everyday_keyword'
    if re.search('科学|物理|植物|实验|学习|阅读', text):
        return 'knowledge_keyword'
    if re.search('记者|会议|发布|财政|企业|市场', text):
        return 'news_keyword'
    return 'other'


def audit(args):
    dest, tokens, prepared = args.output, args.tokens, args.prepared
    report_path = dest / 'data-audit.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    manifest = json.loads((tokens / 'manifest.json').read_text(encoding='utf-8'))
    sp = spm.SentencePieceProcessor(model_proto=(tokens / 'tokenizer.model').read_bytes())
    if checksum(tokens / 'tokenizer.model') != manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer hash mismatch')
    development = json.loads((dest / 'development.json').read_text(encoding='utf-8'))['cases']
    reserved = json.loads((dest / 'reserved-internal.json').read_text(encoding='utf-8'))['cases']
    all_cases = development + reserved
    needles = {c['id']: compact(c['prompt']) for c in all_cases}
    report = {'kind': 'foundation_data_audit_v1', 'splits': {}, 'samples': [],
              'case_matches_in_current_snapshot': [], 'cross_case_exact_or_containment': [],
              'scope': 'Current snapshot train/validation only. Does not certify historical-parent or semantic independence.',
              'human_review_status': 'pending', 'grammar_pass': None}
    for i, left in enumerate(all_cases):
        for right in all_cases[i+1:]:
            if needles[left['id']] in needles[right['id']] or needles[right['id']] in needles[left['id']]:
                report['cross_case_exact_or_containment'].append([left['id'], right['id']])
    for split in ('train', 'validation'):
        data = np.memmap(tokens / f'{split}.bin', mode='r', dtype='<u4')
        eos_chunks, unknown, out_of_range = [], 0, 0
        for start in range(0, len(data), 1000000):
            block = data[start:start+1000000]
            unknown += int(np.count_nonzero(block == sp.unk_id()))
            out_of_range += int(np.count_nonzero(block >= sp.vocab_size()))
            eos_chunks.append(np.flatnonzero(block == sp.eos_id()) + start)
        ends = np.concatenate(eos_chunks)
        sampled, sources, lengths, themes = {}, Counter(), Counter(), Counter()
        blank, ending, with_newlines = 0, 0, 0
        rows = 0
        with (prepared / f'{split}.jsonl').open(encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                text = row['text']
                rows += 1
                source = row['source']
                length = 'short' if len(text) < 600 else 'medium' if len(text) < 1800 else 'long'
                theme = category(text)
                sources[source] += 1
                lengths[length] += 1
                themes[theme] += 1
                blank += int(not text.strip())
                ending += int(bool(re.search(r'[。！？.!?][”’）\)]*$', text.strip())))
                with_newlines += int('\n' in text)
                normalized = compact(text)
                for case_id, prompt in needles.items():
                    if prompt in normalized:
                        report['case_matches_in_current_snapshot'].append({'case_id':case_id, 'split':split, 'id':row['id']})
                key = (source, length, theme)
                priority = hashlib.sha256(('repair-v1:' + row['id']).encode()).hexdigest()
                if key not in sampled or priority < sampled[key][0]:
                    sampled[key] = (priority, index, row)
        if rows != len(ends) or int(ends[-1]) != len(data)-1:
            raise RuntimeError(f'Document/EOS boundary mismatch: {split}')
        sample_errors = []
        for key, (_, index, row) in sorted(sampled.items()):
            begin = 0 if index == 0 else int(ends[index-1])+1
            end = int(ends[index])
            actual = data[begin:end].tolist()
            expected = sp.encode(row['text'])
            decoded = sp.decode(actual)
            same = actual == expected
            same_text = compact(decoded) == compact(row['text'])
            if not same:
                sample_errors.append(row['id'])
            report['samples'].append({'split':split, 'row_index':index, 'id':row['id'],
                'source':row['source'], 'stratum':list(key), 'text':row['text'],
                'decoded':decoded, 'encoded_matches_binary':same,
                'non_whitespace_roundtrip_equal':same_text,
                'unknown_tokens':actual.count(sp.unk_id()),
                'newline_count_before':row['text'].count('\n'), 'newline_count_after':decoded.count('\n'),
                'human_grammar_review':None, 'human_coherence_review':None})
        report['splits'][split] = {'documents':rows, 'tokens':len(data), 'eos_count':len(ends),
            'unknown_tokens':unknown, 'unknown_ratio':unknown/len(data), 'out_of_range':out_of_range,
            'source_counts':dict(sources), 'length_counts':dict(lengths), 'keyword_theme_counts':dict(themes),
            'blank_documents':blank, 'documents_with_terminal_punctuation':ending,
            'documents_with_newlines':with_newlines, 'sample_count':len(sampled),
            'sample_binary_mismatches':sample_errors,
            'input_hash_matches':checksum(prepared / f'{split}.jsonl') == manifest['input_sha256'][split],
            'binary_hash_matches':checksum(tokens / f'{split}.bin') == manifest['binary_sha256'][split]}
        print(f'Audited {split}: {rows} documents, {len(sampled)} stratified samples.', flush=True)
    report['mechanical_pass'] = all(s['out_of_range'] == 0 and not s['sample_binary_mismatches']
        and s['input_hash_matches'] and s['binary_hash_matches'] for s in report['splits'].values())
    report['case_files_sha256'] = {n:checksum(dest/n) for n in ('development.json', 'reserved-internal.json')}
    write(report_path, report)
    lines = ['# 新开发周期：语料分层样本审核', '',
        '以下为既有语料的分层样本，按来源、长度及关键词主题抽取；关键词分类不是语义审核。',
        '只对 train 样本考虑后续训练。validation 样本只用于检查，禁止转作训练材料。',
        '逐条检查语法、段落完整性、指代连贯、隐私、内容适宜性。空白审核字段不是通过。', '']
    for i, row in enumerate(report['samples'], 1):
        lines += [f'## {i:03d} · {row["split"]} · {row["source"]} · {row["id"]}', '',
                  '原文：', '', row['text'], '', '人工结论：待审核', '问题说明：', '']
    (dest/'corpus-review.md').write_text('\n'.join(lines), encoding='utf-8')
    write(dest/'corpus-review-form.json', {'origin':'pending human review; no auto-approval',
        'audit_sha256':checksum(report_path), 'rows':[{'id':s['id'], 'split':s['split'],
        'text_sha256':hashlib.sha256(s['text'].encode()).hexdigest(), 'decision':None,
        'grammar':None, 'coherence':None, 'privacy':None, 'content':None,
        'reviewer':None, 'notes':''} for s in report['samples']]})


def repeat_ratio(text, width=6):
    text = compact(text)
    grams = [text[i:i+width] for i in range(max(0, len(text)-width+1))]
    return 0.0 if not grams else 1 - len(set(grams))/len(grams)


def decode(args):
    dest = args.output
    if (dest/'decoding-results.json').exists():
        raise FileExistsError('Preserve existing diagnostic outputs')
    audit_report = json.loads((dest/'data-audit.json').read_text(encoding='utf-8'))
    if not audit_report['mechanical_pass'] or audit_report['case_matches_in_current_snapshot'] or audit_report['cross_case_exact_or_containment']:
        raise RuntimeError('Fix mechanical checks or overlapping cases before decoding')
    for name, digest in audit_report['case_files_sha256'].items():
        if checksum(dest/name) != digest:
            raise RuntimeError('Case file changed after audit')
    cases = json.loads((dest/'development.json').read_text(encoding='utf-8'))['cases']
    checkpoint_hash = checksum(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['tokenizer_sha256'] != checksum(args.tokens/'tokenizer.model'):
        raise RuntimeError('Tokenizer mismatch')
    sp = spm.SentencePieceProcessor(model_proto=(args.tokens/'tokenizer.model').read_bytes())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PetLM(state['config']).to(device).eval()
    model.load_state_dict(state['model'])
    del state
    variants = [dict(name='original', temperature=0.8, penalty=1.0),
                dict(name='lower_temperature', temperature=0.6, penalty=1.0),
                dict(name='mild_repetition_penalty', temperature=0.8, penalty=1.1)]
    forbidden = [i for i in range(sp.vocab_size()) if i in (sp.unk_id(), sp.bos_id()) or sp.id_to_piece(i).startswith('<0x')]
    result = {'checkpoint_sha256':checkpoint_hash, 'variants':variants, 'seeds':[42,123],
        'max_new_tokens':80, 'top_k':40, 'precision':'float32, same as existing generation',
        'scope':'New development diagnostics only; reserved internal set not evaluated.',
        'penalty_scope':'Each distinct generated token in last 64 generated tokens; excludes prompt.',
        'human_semantic_scores':None, 'outputs':[]}
    with (dest/'decoding-progress.jsonl').open('x', encoding='utf-8') as progress:
        for case in cases:
            for variant in variants:
                for seed in result['seeds']:
                    torch.manual_seed(seed)
                    if device == 'cuda':
                        torch.cuda.manual_seed_all(seed)
                    prompt_ids = sp.encode(case['prompt'])
                    ids, generated = list(prompt_ids), []
                    stopped = False
                    with torch.inference_mode():
                        for _ in range(result['max_new_tokens']):
                            x = torch.tensor([ids[-model.config['context']:]], device=device)
                            logits = model(x)[0,-1].clone()
                            if generated and variant['penalty'] != 1:
                                seen = list(set(generated[-64:]))
                                values = logits[seen]
                                logits[seen] = torch.where(values < 0, values*variant['penalty'], values/variant['penalty'])
                            logits /= variant['temperature']
                            logits[forbidden] = float('-inf')
                            values, indices = logits.topk(40)
                            next_id = indices[torch.multinomial(values.softmax(-1),1)].item()
                            if next_id == sp.eos_id():
                                stopped = True
                                break
                            generated.append(next_id)
                            ids.append(next_id)
                    continuation = sp.decode(generated)
                    row = dict(case, variant=variant['name'], seed=seed, continuation=continuation,
                        raw_generation=sp.decode(ids), generated_token_ids=generated,
                        generated_tokens=len(generated), stopped_on_eos=stopped,
                        repeated_character_6gram_ratio=repeat_ratio(continuation),
                        human_grammar_score=None, human_relevance_score=None)
                    result['outputs'].append(row)
                    progress.write(json.dumps(row, ensure_ascii=False)+'\n')
                    progress.flush()
            print(f'Generated all six comparisons: {case["id"]}', flush=True)
    if checksum(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during diagnostics')
    result['summary'] = {}
    for variant in variants:
        rows = [r for r in result['outputs'] if r['variant']==variant['name']]
        result['summary'][variant['name']] = {'outputs':len(rows),
            'mean_repeat_ratio':sum(r['repeated_character_6gram_ratio'] for r in rows)/len(rows),
            'mean_generated_tokens':sum(r['generated_tokens'] for r in rows)/len(rows),
            'eos_stops':sum(r['stopped_on_eos'] for r in rows)}
    write(dest/'decoding-results.json', result)
    lines = ['# 固定 v26r1 权重的生成对照', '',
        '新开发题，三种设置各两个随机种子。保留所有输出，未运行保留内部题。重复率只衡量字符重复，不是语义或语法评分。', '']
    for row in result['outputs']:
        lines += [f'## {row["id"]} · {row["variant"]} · seed={row["seed"]}', '',
                  row['prompt'], '', row['continuation'], '']
    (dest/'decoding-review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('audit','decode'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, default=Path('data/petlm/tokens-chinese-v26r1-strict-24k'))
    parser.add_argument('--prepared', type=Path, default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    parser.add_argument('--checkpoint', type=Path, default=Path('data/petlm/models/chinese-v26r1-base-adapt-7500/latest.pt'))
    args = parser.parse_args()
    {'audit':audit, 'decode':decode}[args.stage](args)


if __name__ == '__main__':
    main()
