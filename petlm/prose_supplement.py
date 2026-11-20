"""Audit new FineWeb2 batches without training or granting human approval.

Writes additions only. Parent text, splits, tokenizer and model remain unchanged.
Exact/prefix dedup uses the verified parent index; long-paragraph overlap is
checked against every parent pretraining split without displaying its contents.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
from urllib.parse import urlparse

import sentencepiece as spm

from .data import split_for
from .foundation_diagnostics import category, compact
from .pretrain_corpus import clean_document, FINEWEB_SPAM_ORIGIN
from .pretrain_increment import canonical_hashes, PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL
from .prose_repair import reason as prose_reason, hashes as prose_hashes
from .run import checksum


DOMAINS = ('people.com.cn', 'xinhuanet.com', 'news.cn', 'gov.cn', 'edu.cn',
           'cas.cn', 'kepu.net.cn', 'sciencenet.cn', 'thepaper.cn', 'chinadaily.com.cn')
EXTRA_EXCLUSIONS = re.compile('加盟|营收|招商|总部.{0,20}支持|开店|赞助|品牌推广|前列腺|性高潮')


def save(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def build(raw_paths, base, tokenizer_dir, output):
    if output.exists():
        raise FileExistsError(output)
    batches = []
    for raw in raw_paths:
        meta = json.loads(raw.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        if (meta['sha256'] != checksum(raw) or meta['dataset'] != 'HuggingFaceFW/fineweb-2'
                or meta['config'] != 'cmn_Hani' or meta['split'] != 'train'
                or meta['license'] != 'odc-by'):
            raise ValueError(f'Invalid source metadata: {raw}')
        batches.append((raw, meta))
    batches.sort(key=lambda item: item[1]['start_offset'])
    for previous, current in zip(batches, batches[1:]):
        if previous[1]['next_offset'] != current[1]['start_offset']:
            raise ValueError('Input batches must be contiguous and non-overlapping')
    manifest = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    index_meta = json.loads((base/'clean-index.meta.json').read_text(encoding='utf-8'))
    if (index_meta['snapshot_manifest_sha256'] != checksum(base/'manifest.json')
            or index_meta['index_sha256'] != checksum(base/'clean-index.sqlite')):
        raise ValueError('Parent dedup index changed')
    token_meta = json.loads((tokenizer_dir/'manifest.json').read_text(encoding='utf-8'))
    if checksum(tokenizer_dir/'tokenizer.model') != token_meta['tokenizer_sha256']:
        raise ValueError('Parent tokenizer changed')
    sp = spm.SentencePieceProcessor(model_proto=(tokenizer_dir/'tokenizer.model').read_bytes())
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output/'screening-source.py')
    db = sqlite3.connect((base/'clean-index.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    assert db.execute('SELECT count(*) FROM hashes').fetchone()[0] == sum(manifest['splits'].values())
    seen_full, seen_prefix = set(), set()
    rejected, counts, tokens, themes = Counter(), Counter(), Counter(), Counter()
    prose_rejected, pending = Counter(), []
    handles = {s: (output/f'strict-{s}.jsonl').open('x', encoding='utf-8', newline='\n')
               for s in ('train', 'validation', 'test')}
    try:
        for raw, meta in batches:
            row_count, last_index, record_ids = 0, meta['start_offset']-1, set()
            with raw.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    index = row['row_index']
                    assert last_index < index < meta['next_offset']
                    assert (row['dataset'], row['config'], row['split'], row['language'], row['language_script']) == (
                        'HuggingFaceFW/fineweb-2', 'cmn_Hani', 'train', 'cmn', 'Hani')
                    assert row['id'] not in record_ids
                    record_ids.add(row['id'])
                    last_index = index
                    row_count += 1
                    original = row['text']
                    origin = row.get('url', '')
                    if FINEWEB_SPAM_ORIGIN.search(origin):
                        rejected['spam_origin'] += 1
                        continue
                    if any(p.search(original) for p in (PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL)):
                        rejected['raw_personal_or_contact_pattern'] += 1
                        continue
                    text, why, _ = clean_document(original, 'fineweb2', 'strict')
                    if why:
                        rejected[why] += 1
                        continue
                    again, why, _ = clean_document(text, 'fineweb2', 'standard')
                    if why or again != text:
                        rejected['non_idempotent_cleaning'] += 1
                        continue
                    full, prefix = canonical_hashes(text)
                    if (full in seen_full or prefix in seen_prefix or db.execute(
                            'SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1', (full, prefix)).fetchone()):
                        rejected['parent_or_batch_exact_prefix_duplicate'] += 1
                        continue
                    ids = sp.encode(text)
                    if sp.unk_id() in ids or compact(sp.decode(ids)) != compact(text):
                        rejected['tokenizer_roundtrip'] += 1
                        continue
                    seen_full.add(full)
                    seen_prefix.add(prefix)
                    split = split_for(full)
                    item = {'id': full, 'source': 'fineweb2', 'text': text, 'split': split,
                            'origin': origin, 'license': row['license'], 'source_record_id': row['id'],
                            'raw_path': str(raw), 'row_index': index, 'raw_text_sha256': hashlib.sha256(original.encode()).hexdigest(),
                            'tokens_with_eos': len(ids)+1, 'review_status': 'not_human_reviewed'}
                    handles[split].write(json.dumps(item, ensure_ascii=False)+'\n')
                    counts[split] += 1
                    tokens[split] += len(ids)+1
                    themes[f'{split}/{category(text)}'] += 1
                    why = prose_reason(item)
                    if why is None and EXTRA_EXCLUSIONS.search(text):
                        why = 'residual_promotion_or_outside_pilot_scope'
                    if why:
                        prose_rejected[why] += 1
                        continue
                    host = (urlparse(origin).hostname or '').lower()
                    item['within_r3_source_allowlist'] = (any(host == d or host.endswith('.'+d) for d in DOMAINS)
                        and not any(part in ('bbs', 'forum', 'blog') for part in host.split('.')))
                    pending.append(item)
            assert row_count == meta['documents']
            assert last_index+1 == meta['next_offset']
            assert row_count + meta['skipped'] == meta['rows_scanned']
            print(f"Verified rows through {last_index}; strict accepted={sum(counts.values())}; prose candidates={len(pending)}", flush=True)
    finally:
        db.close()
        for handle in handles.values():
            handle.close()
    assert sum(counts.values()) + sum(rejected.values()) == sum(m['documents'] for _, m in batches)
    # Only keep candidate fingerprints in memory; scan parent prose for matching
    # long paragraphs. This does not inspect any frozen dialogue benchmark.
    fingerprints = {h for row in pending for h in prose_hashes(row['text'])}
    overlap = set()
    for split in ('train', 'validation', 'test'):
        path = base/f'{split}.jsonl'
        if checksum(path) != manifest['files'][path.name]:
            raise ValueError(f'Parent {split} changed')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                overlap.update(prose_hashes(json.loads(line)['text']) & fingerprints)
        print(f'Parent {split} paragraph overlap scan complete', flush=True)
    kept, all_seen = [], set()
    for row in sorted(pending, key=lambda r: ({'test': 0, 'validation': 1, 'train': 2}[r['split']], r['id'])):
        hashes = prose_hashes(row['text'])
        if hashes & overlap:
            prose_rejected['parent_exact_prefix_or_long_paragraph_overlap'] += 1
        elif hashes & all_seen:
            prose_rejected['new_candidate_exact_prefix_or_long_paragraph_duplicate'] += 1
        else:
            all_seen.update(hashes)
            kept.append(row)
    candidate_counts, candidate_tokens, candidate_themes = Counter(), Counter(), Counter()
    candidate_handles = {(group, split): (output/f'{group}-{split}.jsonl').open('x', encoding='utf-8', newline='\n')
                         for group in ('source-restricted', 'other-sources-hold') for split in ('train', 'validation', 'test')}
    try:
        for row in kept:
            group = 'source-restricted' if row['within_r3_source_allowlist'] else 'other-sources-hold'
            key = f"{group}/{row['split']}"
            candidate_counts[key] += 1
            candidate_tokens[key] += row['tokens_with_eos']
            candidate_themes[f'{key}/{category(row["text"])}'] += 1
            candidate_handles[group, row['split']].write(json.dumps(row, ensure_ascii=False)+'\n')
    finally:
        for handle in candidate_handles.values():
            handle.close()
    # Every source-restricted train/validation candidate is reviewable in full;
    # other domains remain held, and test prose is omitted from review sheets.
    review = [dict(row, decision=None, reviewer=None, notes='') for row in kept
              if row['within_r3_source_allowlist'] and row['split'] != 'test']
    save(output/'review.json', {'scope': 'New source-restricted training/monitoring candidates only',
                              'status': 'pending_human_review', 'rows': review})
    lines = ['# 新增中文段落审核表', '', '本表未获人工批准；此前 r3 审批不覆盖本批。',
             '来源域名筛选及自动过滤不证明文本自然、内容正确、隐私合规或原网页许可适用。',
             '逐条检查：完整自然、语法与指代连贯、没有广告插词/拼接、内容合适、无个人敏感信息。',
             'validation 仅用于监测，不进入训练。请填写通过/拒绝/待定及理由。', '']
    for i, row in enumerate(review, 1):
        lines += [f'## N{i:03d} · {row["split"]}', f'ID：{row["id"]}',
                  f'来源：{row["origin"]}', '', row['text'], '', '人工结论：待审核', '说明：', '']
    (output/'review.md').write_text('\n'.join(lines), encoding='utf-8')
    report = {'kind': 'new_fineweb_prose_supplement_audit_v1', 'status': 'not_training_ready',
              'parent_snapshot': str(base), 'parent_manifest_sha256': checksum(base/'manifest.json'),
              'tokenizer_sha256': token_meta['tokenizer_sha256'], 'screening_code_sha256': checksum(Path(__file__)),
              'inputs': [dict(path=str(path), **meta) for path, meta in batches],
              'strict_counts': dict(counts), 'strict_tokens_with_eos': dict(tokens),
              'strict_keyword_themes': dict(themes), 'strict_rejections': dict(rejected),
              'candidate_counts': dict(candidate_counts), 'candidate_tokens_with_eos': dict(candidate_tokens),
              'candidate_keyword_themes': dict(candidate_themes), 'prose_rejections': dict(prose_rejected),
              'review_rows': len(review), 'human_review_completed': False, 'training_started': False,
              'physical_actuation': 'disabled', 'limits': [
                  'Keyword categories are heuristics, not human topic or quality judgments.',
                  'No claim of semantic independence or independence from every historical parent.',
                  'Strict additions alone are not approved natural-language repair material.',
                  'Domain allowlist is not original-page license verification; ODC-BY is dataset metadata.',
                  'Text is normalized by existing strict cleaner, not rewritten or synthetically repaired.',
                  'Training still requires sufficient suitable prose, review and fixed limited configuration.'],
              'files': {p.name: checksum(p) for p in output.iterdir() if p.is_file()}}
    save(output/'manifest.json', report)
    print(json.dumps({k: report[k] for k in ('strict_counts', 'strict_tokens_with_eos', 'candidate_counts',
                                           'candidate_tokens_with_eos', 'review_rows')}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, action='append', required=True)
    parser.add_argument('--base', type=Path, default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    parser.add_argument('--tokenizer-dir', type=Path, default=Path('data/petlm/tokens-chinese-v26r1-strict-24k'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.raw, args.base, args.tokenizer_dir, args.output)


if __name__ == '__main__':
    main()
