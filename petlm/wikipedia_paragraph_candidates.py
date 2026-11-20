"""Build attributable, page-isolated Chinese Wikipedia paragraph candidates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
from urllib.parse import quote

import mwparserfromhell
import sentencepiece as spm

from .data import split_for
from .foundation_diagnostics import compact
from .pretrain_increment import canonical_hashes, PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL
from .prose_supplement import save
from .run import checksum
from .wikimedia_prose import (DATE, LICENSE, LICENSE_URL, OUTSIDE_SCOPE,
    RIGHTS_CONCERN, UNSUITABLE_TITLE, fingerprints, normalize, reason)

QUALITY_TAG = re.compile(r'\{\{\s*(?:disputed|accuracy|hoax|copyedit|cleanup|rough translation|expand|unreferenced|refimprove|primary sources|original research|notability|npov|advert|expert|争议|爭議|翻译腔|翻譯腔|需要校对|需要校對|來源請求|来源请求)\b', re.I)
CONTEXT_START = re.compile(r'^(?:他[们們]?|她[们們]?|它[们們]?|牠[们們]?|其[后後]|此[后後处處]|这[些种个]|這[些種個]|该|該|之后|之後|再如|但|同时|同時|第二|第三|首先|其次|最终|最終|随后|隨後|直至|直到|待|责任的)')
CONTEXT_INSIDE = re.compile(r'上面提到|上述提及|前述|下述|前一[节節章]|后一[节節章]|後一[节節章]|这个两个|這個兩個|别性的|別性的|第一层是装饰是|第一層是裝飾是')
LONG_QUOTE = re.compile(r'[“「『][^”」』]{40,}[”」』]')


def paragraphs(raw):
    """Clean each original block separately; refs never join source blocks."""
    section = ''
    for index, original in enumerate(re.split(r'\n\s*\n|(?=^={1,6}[^=])', raw, flags=re.M)):
        block = original.strip()
        heading = re.match(r'^=+\s*(.*?)\s*=+\s*(?:\n|$)', block)
        if heading:
            section = mwparserfromhell.parse(heading[1]).strip_code().strip()
            block = block[heading.end():].strip()
        if not block:
            continue
        block = re.sub(r'<ref\b[^>]*?/\s*>|<ref\b[^>]*>.*?</ref\s*>', '', block, flags=re.I | re.S)
        block = re.sub(r'<!--.*?-->', '', block, flags=re.S)
        if re.search(r'\{\{|\}\}|\{\||\|\}|<|>|-\{|\}-|__|https?://|www\.|^\s*[*#;:=|]|\[\[(?:File|Image|Category|文件|檔案|图像|圖片|分类|分類):', block, re.M | re.I):
            yield index, section, None, 'non_prose_structure_or_external_link', original
            continue
        text = mwparserfromhell.parse(block).strip_code(normalize=True, collapse=False)
        yield index, section, normalize(re.sub(r'\s+', ' ', text)), None, original


def extra_reason(text):
    if CONTEXT_START.search(text) or CONTEXT_INSIDE.search(text):
        return 'missing_context_or_observed_language_defect'
    if LONG_QUOTE.search(text):
        return 'long_third_party_quotation'
    return None


def verified_rows(path, meta):
    count = 0
    seen = set()
    with path.open(encoding='utf-8') as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            if (row.get('url') != meta.get('url')
                    or row.get('eligible_index') != meta['start_offset']+count-1
                    or not isinstance(row.get('raw_wikitext'), str)
                    or not row.get('id') or str(row['id']) in seen):
                raise ValueError('Raw source URL, ordering, page ID, or wikitext changed')
            seen.add(str(row['id']))
            yield count, row
    if count != meta['documents'] or meta['next_offset'] != meta['start_offset']+count:
        raise ValueError('Raw batch count or offsets differ from receipt')


def prior_exclusions(base, parent, roots):
    """Read only metadata from parent provenance and licensed predecessor rows."""
    provenance = base/'provenance.jsonl'
    if checksum(provenance) != parent['files']['provenance.jsonl']:
        raise ValueError('Parent provenance changed')
    page_ids = set()
    with provenance.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('source') == 'wikipedia' and str(row.get('record_id', '')).isdigit():
                page_ids.add(str(row['record_id']))
    prior_hashes, receipts = set(), []
    for root in roots:
        manifest_path = root/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        receipts.append({'path':str(root), 'manifest_sha256':checksum(manifest_path)})
        for name, expected in manifest['files'].items():
            if name.endswith('-candidates.jsonl'):
                path = root/name
                if checksum(path) != expected:
                    raise ValueError('Earlier candidate snapshot changed')
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        prior_hashes.update(fingerprints(json.loads(line)['text']))
    return page_ids, prior_hashes, receipts


def page_split(page_id):
    return split_for(hashlib.sha256(('zhwiki-page:'+str(page_id)).encode()).hexdigest())


def build(raw_paths, base, tokenizer_dir, output, review_limit=120):
    if output.exists():
        raise FileExistsError(output)
    if not raw_paths or review_limit < 1:
        raise ValueError('Raw batches and a positive review limit are required')
    inputs = []
    seen_archives = set()
    for path in raw_paths:
        meta = json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        if (meta.get('project') != 'zhwiki' or meta.get('dump_date') != DATE
                or not meta.get('preserve_wikitext') or meta.get('sha256') != checksum(path)
                or meta.get('archive_file') in seen_archives):
            raise ValueError(f'Unverified, non-markup, or duplicated source batch: {path}')
        seen_archives.add(meta['archive_file'])
        inputs.append((path, meta))

    legal_root = Path('data/petlm/raw/wikimedia-zh-source-r1')
    for name in ('dumps-legal.html', 'cc-by-sa-4.0.html'):
        receipt_path = legal_root/(name+'.receipt.json')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('sha256') != checksum(legal_root/name):
            raise ValueError('Wikimedia license evidence changed')
    if 'creativecommons.org/licenses/by-sa/4.0/' not in (legal_root/'dumps-legal.html').read_text(encoding='utf-8'):
        raise ValueError('Official Wikimedia text license evidence missing')

    parent = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    index_meta = json.loads((base/'clean-index.meta.json').read_text(encoding='utf-8'))
    index_path = base/'clean-index.sqlite'
    if (index_meta['snapshot_manifest_sha256'] != checksum(base/'manifest.json')
            or index_meta['index_sha256'] != checksum(index_path)):
        raise ValueError('Parent dedup index changed')
    tok_meta = json.loads((tokenizer_dir/'manifest.json').read_text(encoding='utf-8'))
    tok_path = tokenizer_dir/'tokenizer.model'
    if checksum(tok_path) != tok_meta['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    tokenizer = spm.SentencePieceProcessor(model_proto=tok_path.read_bytes())
    experiment = Path('data/petlm/experiments/generation-repair-v1')
    parent_pages, prior_hashes, prior_receipts = prior_exclusions(base, parent, [
        experiment/'licensed-prose-reviewed-pilot-r1',
        experiment/'licensed-source-d2l-v1-r1',
        experiment/'licensed-source-expansion-r1'])

    pending, rejected, pages = [], Counter(), Counter()
    db = sqlite3.connect(index_path.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        if db.execute('SELECT count(*) FROM hashes').fetchone()[0] != sum(parent['splits'].values()):
            raise ValueError('Parent dedup index row count mismatch')
        for path, meta in inputs:
            for line_no, record in verified_rows(path, meta):
                    page_id = str(record.get('id') or '')
                    title = str(record.get('title') or '')
                    revision_id = str(record.get('revision_id') or '')
                    raw = record['raw_wikitext']
                    pages[path.name] += 1
                    if page_id in parent_pages:
                        rejected['parent_page_id_overlap'] += 1
                        continue
                    if not page_id or not title or not revision_id:
                        rejected['missing_page_attribution'] += 1
                        continue
                    if UNSUITABLE_TITLE.search(normalize(title)) or RIGHTS_CONCERN.search(raw):
                        rejected['page_scope_or_rights_concern'] += 1
                        continue
                    if QUALITY_TAG.search(raw):
                        rejected['source_page_quality_warning'] += 1
                        continue
                    split = page_split(page_id)
                    for para_idx, section, text, why, original_block in paragraphs(raw):
                        if why:
                            rejected[why] += 1
                            continue
                        if OUTSIDE_SCOPE.search(text):
                            rejected['outside_low_risk_prose_scope'] += 1
                            continue
                        if any(p.search(text) for p in (PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL)):
                            rejected['personal_or_contact_pattern'] += 1
                            continue
                        why = reason(text)
                        why = why or extra_reason(text)
                        if why:
                            rejected[why] += 1
                            continue
                        full, prefix = canonical_hashes(text)
                        if fingerprints(text) & prior_hashes:
                            rejected['previous_licensed_candidate_overlap'] += 1
                            continue
                        if db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1', (full, prefix)).fetchone():
                            rejected['parent_exact_or_prefix_duplicate'] += 1
                            continue
                        ids = tokenizer.encode(text)
                        if tokenizer.unk_id() in ids or compact(tokenizer.decode(ids)) != compact(text):
                            rejected['tokenizer_roundtrip'] += 1
                            continue
                        origin = ('https://zh.wikipedia.org/w/index.php?title='
                                  + quote(title, safe='')+'&oldid='+quote(revision_id, safe=''))
                        pending.append({'id': full, 'source': 'wikipedia', 'text': text,
                            'title': normalize(title), 'original_title': title,
                            'section': normalize(section), 'split': split,
                            'split_group': 'zhwiki-page:'+page_id,
                            'origin': origin, 'history_url':
                                'https://zh.wikipedia.org/w/index.php?title='+quote(title, safe='')+'&action=history',
                            'attribution': f'{title} — 中文维基百科贡献者',
                            'license': LICENSE, 'license_url': LICENSE_URL,
                            'raw_dump_url': meta['url'], 'archive_file': meta['archive_file'],
                            'page_id': page_id, 'revision_id': revision_id,
                            'revision_timestamp': record.get('timestamp'),
                            'paragraph_index': para_idx,
                            'raw_page_text_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                            'raw_paragraph_sha256': hashlib.sha256(original_block.encode()).hexdigest(),
                            'transformations': ['removed ref elements and HTML comments',
                                'removed MediaWiki link/formatting syntax',
                                'NFKC normalization', 'traditional-to-simplified conversion',
                                'collapsed paragraph whitespace'],
                            'tokens_with_eos': len(ids)+1,
                            'human_review_status': 'pending', 'training_eligible': False})
                    if line_no % 5000 == 0:
                        print(f'{path.name}: {line_no} pages inspected; {len(pending)} paragraph candidates.', flush=True)
    finally:
        db.close()

    wanted = set().union(*(fingerprints(row['text']) for row in pending)) if pending else set()
    parent_seen = set()
    for split in ('train', 'validation', 'test'):
        parent_path = base/(split+'.jsonl')
        if checksum(parent_path) != parent['files'][parent_path.name]:
            raise ValueError(f'Parent {split} data changed')
        with parent_path.open(encoding='utf-8') as stream:
            for line in stream:
                parent_seen.update(fingerprints(json.loads(line)['text']) & wanted)
        print(f'Parent {split}: paragraph and sentence overlap scan complete.', flush=True)

    kept, duplicate_seen = [], set()
    for row in sorted(pending, key=lambda r: ({'test': 0, 'validation': 1, 'train': 2}[r['split']], r['id'])):
        hashes = fingerprints(row['text'])
        if hashes & parent_seen:
            rejected['parent_paragraph_or_sentence_overlap'] += 1
        elif hashes & duplicate_seen:
            rejected['candidate_paragraph_or_sentence_duplicate'] += 1
        else:
            duplicate_seen.update(hashes)
            kept.append(row)

    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output/'screening-source.py')
    counts, tokens, groups = Counter(), Counter(), {}
    for split in ('train', 'validation', 'test'):
        with (output/(split+'-candidates.jsonl')).open('x', encoding='utf-8', newline='\n') as handle:
            for row in kept:
                if row['split'] == split:
                    if groups.setdefault(row['split_group'], split) != split:
                        raise ValueError('Page group leaked across partitions')
                    counts[split] += 1
                    tokens[split] += row['tokens_with_eos']
                    handle.write(json.dumps(row, ensure_ascii=False)+'\n')

    review = [row for row in kept if row['split'] in ('train', 'validation')]
    review.sort(key=lambda r: (r['split'], r['id']))
    all_review = list(review)
    if len(review) > review_limit:
        train = [r for r in review if r['split'] == 'train']
        val = [r for r in review if r['split'] == 'validation']
        train_n = max(0, review_limit-min(len(val), max(1, review_limit//10)))
        review = train[:train_n] + val[:review_limit-train_n]
    save(output/'sample-review.json', {'status':'assistant_sample_inspection_pending',
        'scope':'Sample of training and validation candidates; not human approval.', 'rows':review})
    save(output/'review.json', {'status':'pending_new_human_review',
        'scope':'All train and validation paragraphs in the indexed review batches; test excluded.',
        'rows':[dict(r, decision=None, reviewer=None, notes='') for r in all_review]})
    lines = ['# 新增中文维基百科段落审核表', '',
        '来源：固定日期中文维基百科转储，文本许可 CC BY-SA 4.0；每条保留永久修订链接。',
        '本批是新候选，既往审核不覆盖。逐条检查自然度、完整性、上下文、事实风险、隐私及版权例外。',
        '下列分批文件包含全部 train/validation 待审行；test 不在审核表中。',
        '引用标记和格式被移除；文本未改写或跨段拼接。测试分区不在本表中。', '']
    batch_dir = output/'review-batches'
    batch_dir.mkdir()
    for start in range(0, len(all_review), 100):
        batch_lines = [f'# 人工审核批次 {start//100+1:03d}', '',
            '每条检查：自然完整、上下文独立、事实与建议适宜、无隐私风险、署名与许可无例外。',
            '填写通过/拒绝/待定、审核人和理由；自动筛选不是人工结论。', '']
        for i, row in enumerate(all_review[start:start+100], start+1):
            batch_lines += [f'## W{i:04d} · {row["split"]} · {row["title"]}',
            f'ID：{row["id"]}', f'来源：{row["origin"]}',
            f'作者历史：{row["history_url"]}', f'许可：{row["license"]} · {row["license_url"]}',
            '', row['text'], '', '人工结论：待审核', '审核人：', '理由：', '']
        name = f'batch-{start//100+1:03d}.md'
        (batch_dir/name).write_text('\n'.join(batch_lines), encoding='utf-8')
        lines.append(f'- [批次 {start//100+1:03d}：W{start+1:04d}–W{min(start+100,len(all_review)):04d}](review-batches/{name})')
    (output/'review.md').write_text('\n'.join(lines), encoding='utf-8')

    result = {'kind':'wikipedia_original_paragraph_candidates_v1',
        'status':'pending_human_review_not_training_ready',
        'sources':[{'path':str(p), **m, 'license':LICENSE, 'license_url':LICENSE_URL} for p,m in inputs],
        'parent_manifest_sha256':checksum(base/'manifest.json'),
        'parent_index_sha256':checksum(index_path), 'tokenizer_sha256':tok_meta['tokenizer_sha256'],
        'prior_candidate_exclusions':prior_receipts,
        'parent_wikipedia_page_ids':len(parent_pages),
        'helper_code_sha256':{str(p):checksum(p) for p in [Path('petlm/wikimedia_prose.py'), Path('petlm/pretrain_corpus.py')]},
        'pages_inspected':dict(pages), 'candidate_counts':dict(counts),
        'candidate_tokens_with_eos':dict(tokens), 'review_rows':len(all_review), 'assistant_sample_rows':len(review),
        'rejected':dict(rejected),
        'split_policy':'Stable hash of source page ID; all paragraphs of each page stay together.',
        'dedup_policy':'Exact/prefix parent index plus full, prefix, intact-paragraph and long-sentence fingerprints against every parent partition and candidate partition.',
        'limitations':['Automated screening does not replace human quality review.',
            'All train/validation rows require review; sample inspection is not approval.',
            'Held-out source pages are independent of training pages but are not the frozen blind benchmark.',
            'Deduplication cannot establish semantic novelty.',
            'No training was started; data scale and independent validation gates remain.'],
        'human_review_completed':False, 'training_ready':False,
        'training_started':False, 'physical_actuation':'disabled',
        'old_frozen_benchmarks_used':False,
        'files':{p.relative_to(output).as_posix():checksum(p) for p in output.rglob('*') if p.is_file()}}
    save(output/'manifest.json', result)
    print(json.dumps({'candidate_counts':dict(counts), 'candidate_tokens_with_eos':dict(tokens),
        'review_rows':len(all_review), 'rejected':dict(rejected)}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, action='append', required=True)
    parser.add_argument('--base', type=Path, default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    parser.add_argument('--tokenizer-dir', type=Path, default=Path('data/petlm/tokens-chinese-v26r1-strict-24k'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-limit', type=int, default=120)
    args = parser.parse_args()
    build(args.raw, args.base, args.tokenizer_dir, args.output, args.review_limit)


if __name__ == '__main__':
    main()
