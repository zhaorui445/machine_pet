"""Build attributed Chinese Wikipedia prose candidates; never approve or train."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
from urllib.parse import quote

import sentencepiece as spm

from .data import split_for
from .foundation_diagnostics import compact
from .pretrain_corpus import clean_document
from .pretrain_increment import canonical_hashes, PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL
from .prose_repair import reason as prose_reason
from .wikimedia_prose import fingerprints as prose_hashes, RIGHTS_CONCERN
from .prose_supplement import save
from .run import checksum

LICENSE = 'CC-BY-SA-4.0'
LICENSE_URL = 'https://creativecommons.org/licenses/by-sa/4.0/'


def article_split(page_id):
    return split_for(hashlib.sha256(('zhwiki-page:'+str(page_id)).encode()).hexdigest())


def build(raw_paths, base, tokenizer_dir, output, review_limit=40):
    if output.exists():
        raise FileExistsError(output)
    if not raw_paths:
        raise ValueError('At least one raw source batch is required')
    if review_limit < 1:
        raise ValueError('Review sample size must be positive')
    batches = []
    for raw in raw_paths:
        metadata_path = raw.with_suffix('.metadata.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if (metadata.get('project') != 'zhwiki' or metadata.get('dump_date') != '20261101'
                or not str(metadata.get('archive_file', '')).startswith('zhwiki-20261101-pages-articles')
                or metadata.get('sha256') != checksum(raw)):
            raise ValueError(f'Raw input is not a verified fixed-date Wikipedia segment: {raw}')
        batches.append((raw, metadata))
    legal_root = Path('data/petlm/raw/wikimedia-zh-source-r1')
    legal = (legal_root/'dumps-legal.html').read_text(encoding='utf-8')
    if 'creativecommons.org/licenses/by-sa/4.0/' not in legal:
        raise ValueError('Official Wikimedia text-license evidence is missing')
    for evidence_name in ('dumps-legal.html', 'cc-by-sa-4.0.html'):
        receipt = json.loads((legal_root/(evidence_name+'.receipt.json')).read_text(encoding='utf-8'))
        if receipt.get('sha256') != checksum(legal_root/evidence_name):
            raise ValueError('Wikimedia license evidence changed')

    parent = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    index_meta = json.loads((base/'clean-index.meta.json').read_text(encoding='utf-8'))
    if (index_meta['snapshot_manifest_sha256'] != checksum(base/'manifest.json')
            or index_meta['index_sha256'] != checksum(base/'clean-index.sqlite')):
        raise ValueError('Parent dedup index changed')
    tokenizer_meta = json.loads((tokenizer_dir/'manifest.json').read_text(encoding='utf-8'))
    tokenizer_path = tokenizer_dir/'tokenizer.model'
    if checksum(tokenizer_path) != tokenizer_meta['tokenizer_sha256']:
        raise ValueError('Parent tokenizer changed')
    tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())

    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output/'screening-source.py')
    pending, rejected, seen_ids = [], Counter(), set()
    db = sqlite3.connect((base/'clean-index.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        assert db.execute('SELECT count(*) FROM hashes').fetchone()[0] == sum(parent['splits'].values())
        for raw, metadata in batches:
          with raw.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                identifier = str(row.get('id') or '')
                if not identifier or identifier in seen_ids or not isinstance(row.get('text'), str):
                    rejected['invalid_or_duplicate_source_row'] += 1
                    continue
                seen_ids.add(identifier)
                if (row.get('url') != metadata['url'] or
                        row.get('eligible_index') != metadata['start_offset']+line_number-1):
                    raise ValueError('Raw source URL or sequential index changed')
                original = row['text']
                if RIGHTS_CONCERN.search(original):
                    rejected['visible_rights_concern'] += 1
                    continue
                if any(pattern.search(original) for pattern in (PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL)):
                    rejected['personal_or_contact_pattern'] += 1
                    continue
                text, why, _ = clean_document(original, 'wikipedia', 'strict')
                if why:
                    rejected[f'clean/{why}'] += 1
                    continue
                candidate = {'id': hashlib.sha256(compact(text).encode()).hexdigest(),
                             'source': 'wikipedia', 'text': text}
                why = prose_reason(candidate)
                if why:
                    rejected[f'prose/{why}'] += 1
                    continue
                full, prefix = canonical_hashes(text)
                if db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1',
                              (full, prefix)).fetchone():
                    rejected['parent_exact_or_prefix_duplicate'] += 1
                    continue
                ids = tokenizer.encode(text)
                if tokenizer.unk_id() in ids or compact(tokenizer.decode(ids)) != compact(text):
                    rejected['tokenizer_roundtrip'] += 1
                    continue
                title = str(row.get('title') or '')
                revision_id = str(row.get('revision_id') or '')
                if not title or not revision_id:
                    rejected['missing_page_attribution'] += 1
                    continue
                origin = ('https://zh.wikipedia.org/w/index.php?title='
                          + quote(title, safe='')+'&oldid='+quote(revision_id, safe=''))
                pending.append({'id': candidate['id'], 'source': 'wikipedia', 'text': text,
                    'title': title, 'revision_id': revision_id, 'timestamp': row.get('timestamp'),
                    'split': article_split(identifier), 'split_group': 'zhwiki-page:'+identifier, 'origin': origin,
                    'history_url': 'https://zh.wikipedia.org/w/index.php?title='
                                   +quote(title, safe='')+'&action=history',
                    'attribution': f'{title} — 中文维基百科贡献者',
                    'license': LICENSE, 'license_url': LICENSE_URL,
                    'raw_dump_url': row['url'], 'raw_document_id': identifier,
                    'raw_text_sha256': hashlib.sha256(original.encode()).hexdigest(),
                    'tokens_with_eos': len(ids)+1, 'human_review_status': 'pending',
                    'training_eligible': False, 'reviewer': None, 'human_decision': None})
                if line_number % 2000 == 0:
                    print(f'Inspected {line_number} rows from {raw.name}; {len(pending)} remain candidates.', flush=True)
            if line_number != metadata['documents'] or metadata['next_offset'] != metadata['start_offset']+line_number:
                raise ValueError('Raw row count or offset range differs from metadata')
    finally:
        db.close()

    wanted = set().union(*(prose_hashes(row['text']) for row in pending)) if pending else set()
    parent_overlap = set()
    for split in ('train', 'validation', 'test'):
        path = base/f'{split}.jsonl'
        if checksum(path) != parent['files'][path.name]:
            raise ValueError(f'Parent {split} data changed')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                parent_overlap.update(prose_hashes(json.loads(line)['text']) & wanted)
        print(f'Parent {split}: sentence and paragraph overlap scan complete.', flush=True)

    kept, seen = [], set()
    for row in sorted(pending, key=lambda item: ({'test': 0, 'validation': 1, 'train': 2}[item['split']], item['id'])):
        hashes = prose_hashes(row['text'])
        if hashes & parent_overlap:
            rejected['parent_sentence_or_paragraph_overlap'] += 1
        elif hashes & seen:
            rejected['candidate_cross_split_or_exact_duplicate'] += 1
        else:
            seen.update(hashes)
            kept.append(row)

    counts, tokens, groups = Counter(), Counter(), {}
    for split in ('train', 'validation', 'test'):
        with (output/f'{split}-candidates.jsonl').open('x', encoding='utf-8', newline='\n') as handle:
            for row in kept:
                if row['split'] != split:
                    continue
                if groups.setdefault(row['split_group'], split) != split:
                    raise ValueError('Article group spans partitions')
                counts[split] += 1
                tokens[split] += row['tokens_with_eos']
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')

    review = [row for row in kept if row['split'] in ('train', 'validation')]
    review.sort(key=lambda row: row['id'])
    sample_train = [row for row in review if row['split'] == 'train'][:max(1, review_limit-8)]
    sample_val = [row for row in review if row['split'] == 'validation'][:min(8, review_limit-len(sample_train))]
    review_sample = sample_train+sample_val
    save(output/'review.json', {'status':'pending_new_human_sample_review',
        'scope':'Deterministic sample only; approval does not cover all corpus rows',
        'candidate_counts':dict(counts), 'rows':review_sample})
    lines = ['# 新增中文维基百科文本抽样审核', '',
        '来源为固定日期的官方中文维基百科转储，正文许可 CC BY-SA 4.0。每条保留永久修订链接。',
        '本表是新来源抽样；逐条检查中文自然度、完整性、事实、个人信息和版权例外。',
        '自动筛选和抽样通过不等于全量语料人工逐条通过。测试分区不在本表中。',
        '如判定不合格，请注明行号与原因；不能修改或用测试分区替换候选。', '']
    for index, row in enumerate(review_sample, 1):
        lines += [f'## W{index:03d} · {row["split"]} · {row["title"]}',
            f'ID：{row["id"]}', f'来源：{row["origin"]}', f'作者历史：{row["history_url"]}',
            f'署名：{row["attribution"]}', f'许可：{row["license"]} · {row["license_url"]}', '',
            row['text'], '', '人工结论：待审核', '审核人：', '理由：', '']
    (output/'review.md').write_text('\n'.join(lines), encoding='utf-8')
    report = {'kind':'licensed_wikipedia_prose_candidate_v1', 'status':'pending_human_sample_review',
        'sources':[{'project':metadata['project'], 'dump_date':metadata['dump_date'],
            'archive_file':metadata['archive_file'], 'url':metadata['url'],
            'raw_sha256':metadata['sha256'], 'documents':metadata['documents'],
            'path':str(raw), 'license':LICENSE, 'license_url':LICENSE_URL}
            for raw, metadata in batches],
        'parent_manifest_sha256':checksum(base/'manifest.json'),
        'parent_index_sha256':checksum(base/'clean-index.sqlite'),
        'tokenizer_sha256':tokenizer_meta['tokenizer_sha256'],
        'counts':dict(counts), 'tokens_with_eos':dict(tokens),
        'screened_candidates':len(pending), 'review_sample_rows':len(review_sample),
        'rejected':dict(rejected), 'split_policy':'Stable hash of zhwiki page ID; every page stays in one partition.',
        'dedup_policy':'Parent exact/prefix index plus full/prefix/paragraph/sentence fingerprints across all parent splits and candidate partitions.',
        'human_review_completed':False, 'training_ready':False, 'training_started':False,
        'physical_actuation':'disabled', 'old_frozen_benchmarks_used':False,
        'limitations':['Wikipedia is encyclopedic prose, not everyday dialogue.',
            'Exact and sentence-level dedup do not prove semantic novelty.',
            'Upstream extraction stripped wiki markup; per-page rights exceptions require source review. Candidate status is not a complete rights clearance.',
            'A sample review is not full-corpus item-by-item approval.',
            'A held-out article split is not the project frozen blind benchmark.',
            'This batch alone does not authorize or establish sufficient training scale.'],
        'files':{path.name:checksum(path) for path in output.iterdir() if path.is_file()}}
    save(output/'manifest.json', report)
    print(json.dumps({'counts':dict(counts), 'tokens_with_eos':dict(tokens),
        'review_sample_rows':len(review_sample), 'rejected':dict(rejected)}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, action='append', required=True)
    parser.add_argument('--base', type=Path, default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    parser.add_argument('--tokenizer-dir', type=Path, default=Path('data/petlm/tokens-chinese-v26r1-strict-24k'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-limit', type=int, default=40)
    args = parser.parse_args()
    build(args.raw, args.base, args.tokenizer_dir, args.output, args.review_limit)


if __name__ == '__main__':
    main()
