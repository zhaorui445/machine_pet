"""Prepare a bounded, deduplicated author-blog review batch for foundation data."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from petlm.blog_prose import extract_html, quality_reason
from petlm.foundation_diagnostics import compact
from petlm.pretrain_increment import canonical_hashes
from petlm.run import checksum
from petlm.wikimedia_prose import fingerprints, normalize

D = Path(__file__).resolve().parent
CYCLE = D.parent
DATA = ROOT / 'data' / 'petlm'
RAW = DATA / 'raw' / 'author-blogs-r9'
PARENT = DATA / 'prepared-chinese-v26r1-strict'
TOK = DATA / 'tokens-chinese-v26r1-strict-24k'
R6 = CYCLE / 'non-fineweb2-bulk-intake-r6'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def markdown_blocks(text):
    section = ''
    index = 0
    in_fence = False
    current = []
    for line in text.splitlines() + ['']:
        if re.match(r'^\s*(```|~~~)', line):
            in_fence = not in_fence
            current = []
            continue
        if in_fence:
            continue
        heading = re.match(r'^#{1,6}\s+(.+?)\s*#*\s*$', line)
        if heading:
            current = []
            section = heading.group(1).strip()
            index += 1
            continue
        if line.strip():
            current.append(line.strip())
            continue
        if current:
            index += 1
            value = ' '.join(current)
            value = re.sub(r'!?\[([^\]]+)\]\([^)]*\)', r'\1', value)
            value = re.sub(r'[`*_~]', '', value)
            value = re.sub(r'<[^>]+>', '', value).strip()
            yield index, section, value
            current = []


def ngrams(text, n=5):
    return {text[i:i+n] for i in range(len(text)-n+1)}


def norm_for_near(text):
    return re.sub(r'[^\w]', '', normalize(text)).lower()


def similar(a, b):
    a, b = ngrams(norm_for_near(a)), ngrams(norm_for_near(b))
    return bool(len(a) >= 70 and len(b) >= 70 and len(a & b) / min(len(a), len(b)) >= .78)


def main():
    prior = load(R6 / 'source-partition.json')
    plan = load(CYCLE / 'acquisition-plan-r1.json')
    assert prior['acquisition_plan_sha256'] == checksum(CYCLE / 'acquisition-plan-r1.json')
    assert 'fineweb2' in prior['excluded']
    assert not prior.get('training_allowed') and not prior.get('sft_allowed')
    assert checksum(PARENT / 'manifest.json') == prior['parent_manifest_sha256']
    tok_sha = checksum(TOK / 'tokenizer.model')
    assert tok_sha == prior['tokenizer_sha256']
    assert checksum(RAW / 'source-policy.json') == prior['source_policy_sha256']
    partition = dict(prior)
    partition['kind'] = 'author_blog_r9_candidate_partition_r2'
    partition['status'] = 'fixed_before_r7_screening'
    partition['recorded_date'] = datetime.now(timezone.utc).date().isoformat()
    partition['selection'] = {
        'ideal': ['596.html'],
        'antfu': ['about-yak-shaving-zh.md', 'binfe-2020-zh.md'],
        'liam': ['five-step-training-method-for-English-listening.html',
                 'pitch-interval-and-harmonic.html', 'Monty-Hall-problem.html',
                 'figures-in-scitech-papers.html', 'tide-influence-on-the-moon.html'],
    }
    partition['rule'] = ('Author-level split fixed before screening. R7 may use only new complete paragraphs '
                         'from these already-fixed documents; same-article passage reuse is allowed, but exact, '
                         'sentence, parent and near-duplicate checks remain mandatory.')
    save(D / 'source-partition.json', partition)

    # Reuse the audited history collector from R6; R6 itself is included as historical input.
    spec = importlib.util.spec_from_file_location('r6_screen', R6 / 'screen.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reviewed, historical, historical_docs, historical_texts = mod.history()
    r6_review = load(R6 / 'review.json')
    reviewed.add(R6 / 'review.json')
    for row in r6_review['rows']:
        historical_texts.append(row['text'])
        historical.update(fingerprints(normalize(row['text'])))
    # Historical review rows sometimes bundle several source paragraphs; index each one too.
    for value in historical_texts:
        for paragraph in re.split(r'\n+', value):
            if paragraph.strip():
                historical.update(fingerprints(normalize(paragraph)))

    parent = load(PARENT / 'manifest.json')
    sp = spm.SentencePieceProcessor(model_proto=(TOK / 'tokenizer.model').read_bytes())
    import sqlite3
    db = sqlite3.connect((PARENT / 'clean-index.sqlite').as_uri() + '?mode=ro', uri=True)
    candidates, rejects = [], Counter()

    def add(source, topic, split, text, title, origin, attribution, license_name,
            license_url, raw_path, raw_sha, section, paragraph_indices):
        text = normalize(text)
        why = quality_reason(text)
        if not why and re.search(r'在此下载|见.{0,12}一节|还是之前的例子|之前的例子|我们之前提到|我猜大家多多少少都有过类似的经历|这张.{0,8}(?:图|GIF)|这幅图|如图所示|^而希望|EVA.{0,8}渚薰|写作的缘和由', text):
            why = 'depends_on_omitted_context'
        if why:
            rejects[f'{source}/{why}'] += 1
            return
        full, prefix = canonical_hashes(text)
        if db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1', (full, prefix)).fetchone():
            rejects[f'{source}/parent_index_exact_or_prefix'] += 1
            return
        if fingerprints(text) & historical:
            rejects[f'{source}/historical_exact_prefix_sentence'] += 1
            return
        ids = sp.encode(text)
        if sp.unk_id() in ids or compact(sp.decode(ids)) != compact(text):
            rejects[f'{source}/tokenizer_roundtrip'] += 1
            return
        candidates.append(dict(id=full, source=source, topic=topic, split=split, text=text,
                               title=title, origin=origin, attribution=attribution,
                               license=license_name, license_url=license_url,
                               raw_payload_sha256=raw_sha,
                               receipt_path=str(raw_path.relative_to(ROOT)).replace('\\', '/') + '.receipt.json',
                               section=section, original_paragraph_indices=paragraph_indices,
                               tokens_with_eos=len(ids)+1, human_decision=None,
                               training_eligible=False, validation_admitted=False))

    policy = load(RAW / 'source-policy.json')
    specs = [
        ('liam', 'five-step-training-method-for-English-listening.html', 'author_blog_liam', '学习与表达', 'validation', 'https://liam.page/2020/03/29/five-step-training-method-for-English-listening', 'Liam Huang', 'CC-BY-SA-4.0', 'https://creativecommons.org/licenses/by-sa/4.0/'),
        ('liam', 'pitch-interval-and-harmonic.html', 'author_blog_liam', '科学与艺术', 'validation', 'https://liam.page/2018/04/09/pitch-interval-and-harmonic', 'Liam Huang', 'CC-BY-SA-4.0', 'https://creativecommons.org/licenses/by-sa/4.0/'),
        ('liam', 'Monty-Hall-problem.html', 'author_blog_liam', '数学与科学', 'validation', 'https://liam.page/2018/02/26/Monty-Hall-problem', 'Liam Huang', 'CC-BY-SA-4.0', 'https://creativecommons.org/licenses/by-sa/4.0/'),
        ('liam', 'figures-in-scitech-papers.html', 'author_blog_liam', '科学与表达', 'validation', 'https://liam.page/2019/07/30/figures-in-scitech-papers', 'Liam Huang', 'CC-BY-SA-4.0', 'https://creativecommons.org/licenses/by-sa/4.0/'),
        ('liam', 'tide-influence-on-the-moon.html', 'author_blog_liam', '科学与自然', 'validation', 'https://liam.page/2016/09/28/tide-influence-on-the-moon', 'Liam Huang', 'CC-BY-SA-4.0', 'https://creativecommons.org/licenses/by-sa/4.0/'),
        ('ideal', '596.html', 'author_blog_ideal', '学习与技术', 'train', 'https://idealclover.top/archives/596', 'idealclover', 'CC-BY-NC-SA-4.0', 'https://creativecommons.org/licenses/by-nc-sa/4.0/'),
    ]
    allowed = set()
    for author, filename, source, topic, split, origin, attribution, lic, lic_url in specs:
        # The source partition's role, not the topic or paragraph text, decides the split.
        assert split == ('validation' if author == 'liam' else 'train')
        allowed.add((author, filename))
        path = RAW / author / filename
        receipt_path = path.with_name(path.name + '.receipt.json')
        receipt = load(receipt_path)
        raw_sha = checksum(path)
        assert raw_sha == receipt['sha256'] and path.stat().st_size == receipt['bytes']
        if path.suffix == '.html':
            doc = extract_html(path.read_text(encoding='utf-8'))
            blocks = [(r['paragraph_index'], r['section'], r['original_text'], r['structural_rejection']) for r in doc['paragraphs']]
            title = doc['title']
        else:
            blocks = [(i, sec, value, None) for i, sec, value in markdown_blocks(path.read_text(encoding='utf-8'))]
            title = path.stem.replace('-', ' ')
        for ix, section, text, structural in blocks:
            if structural:
                rejects[f'{source}/{structural}'] += 1
                continue
            add(source, topic, split, text, title, origin, attribution, lic, lic_url,
                path, raw_sha, section, [ix])

    antfu = policy['sources']['antfu']
    commit = antfu['commit']
    for filename, title, topic in [
        ('about-yak-shaving-zh.md', '关于 Yak Shaving', '学习与技术'),
        ('binfe-2020-zh.md', 'import { reactive } from \'vue\'', '教育与技术'),
    ]:
        path = RAW / 'antfu' / filename
        receipt = load(path.with_name(path.name + '.receipt.json'))
        raw_sha = checksum(path)
        assert raw_sha == receipt['sha256'] and path.stat().st_size == receipt['bytes']
        origin = f'https://raw.githubusercontent.com/antfu/antfu.me/{commit}/pages/posts/{filename}'
        for ix, section, text in markdown_blocks(path.read_text(encoding='utf-8')):
            add('author_blog_antfu', topic, 'train', text, title, origin, 'Anthony Fu',
                antfu['license'], antfu['license_url'], path, raw_sha, section, [ix])

    db.close()
    # Confirm candidate near-duplicates against the parent corpus using 8-CJK probes.
    needles = set()
    for r in candidates:
        for seq in re.findall(r'[\u3400-\u9fff]{12,}', r['text']):
            needles.update(seq[i:i+12] for i in range(0, len(seq)-11, 8))
    probe_file = D / 'overlap-probes.txt'
    probe_file.write_text('\n'.join(sorted(needles)) + '\n', encoding='utf-8', newline='\n')
    wanted = set().union(*(fingerprints(r['text']) for r in candidates)) if candidates else set()
    parent_exact = set()
    splits = [PARENT / f'{s}.jsonl' for s in ('train', 'validation', 'test')]
    for file in splits:
        for line in file.open(encoding='utf-8-sig'):
            row = json.loads(line)
            parent_exact.update(fingerprints(row['text']) & wanted)
    possible_parent = set()
    if needles:
        proc = subprocess.run(['rg', '--no-config', '-F', '-f', str(probe_file), '--no-filename', '--', *map(str, splits)],
                              capture_output=True, text=True, encoding='utf-8', errors='replace')
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr)
        for line in proc.stdout.splitlines():
            try:
                target = json.loads(line)['text']
            except (json.JSONDecodeError, KeyError):
                continue
            for i, candidate in enumerate(candidates):
                if similar(candidate['text'], target):
                    possible_parent.add(i)

    history_near = set()
    candidate_by_probe = {}
    for i, candidate in enumerate(candidates):
        if fingerprints(candidate['text']) & historical:
            continue
        for seq in re.findall(r'[\u3400-\u9fff]{12,}', candidate['text']):
            for j in range(0, len(seq)-11, 8):
                candidate_by_probe.setdefault(seq[j:j+12], set()).add(i)
    for previous in historical_texts:
        possible = set()
        for seq in re.findall(r'[\u3400-\u9fff]{12,}', previous):
            for j in range(len(seq)-11):
                possible.update(candidate_by_probe.get(seq[j:j+12], ()))
        for i in possible - history_near:
            if similar(candidates[i]['text'], previous):
                history_near.add(i)
    selected, accepted_texts, seen_candidates, per_doc, per_source = [], [], set(), Counter(), Counter()
    for i, row in enumerate(candidates):
        reason = ('parent_exact_prefix_paragraph_or_sentence' if fingerprints(row['text']) & parent_exact else
                  'parent_near_duplicate' if i in possible_parent else
                  'historical_review_near_duplicate' if i in history_near else None)
        row_fingerprints = fingerprints(row['text'])
        if not reason and (row_fingerprints & seen_candidates or any(similar(row['text'], t) for t in accepted_texts)):
            reason = 'candidate_duplicate'
        if reason:
            rejects[f'{row["source"]}/{reason}'] += 1
            continue
        doc_key = (row['source'], row['title'])
        if per_doc[doc_key] >= 2 or len(selected) >= 16:
            rejects[f'{row["source"]}/batch_bound'] += 1
            continue
        per_doc[doc_key] += 1
        per_source[row['source']] += 1
        selected.append(row)
        accepted_texts.append(row['text'])
        seen_candidates.update(row_fingerprints)
    if not selected:
        raise RuntimeError('No candidates remain after screening; no review batch was created.')

    save(D / 'review.json', dict(status='pending_human_review', rows=selected))
    audit = dict(kind='r7_pre_review_quality_and_dedup_audit', historical_review_files=len(reviewed),
                 candidate_rows_before_dedup=len(candidates), selected_rows=len(selected),
                 rejections=dict(rejects), near_duplicate_method='normalized CJK character 5-grams; 78% containment; parent searched by 8-CJK probes',
                 caveat='Approximate deduplication does not establish semantic originality; reviewer should inspect paraphrases and source context.')
    save(D / 'pre-review-quality-audit.json', audit)
    counts = Counter()
    for row in selected:
        counts[(row['source'], row['split'], row['topic'])] += 1
    lines = ['# 中文正文审核（FineWeb2 已排除）', '',
             '本批仅供人工审核。请核对自然度、完整性、隐私、来源署名、许可和重复情况。训练与验证仍按作者级分区。', '',
             '| 来源 | 用途 | 主题 | 条数 | Tokens（含结束符） |', '| --- | --- | --- | ---: | ---: |']
    for (source, split, topic), n in counts.items():
        tokens = sum(r['tokens_with_eos'] for r in selected if (r['source'], r['split'], r['topic']) == (source, split, topic))
        lines.append(f'| {source} | {"独立验证候选" if split == "validation" else "训练候选"} | {topic} | {n} | {tokens} |')
    for i, row in enumerate(selected, 1):
        lines += ['', f'## {i}. {row["title"]}', '', f'用途：{"独立验证候选" if row["split"] == "validation" else "训练候选"}；段落编号：{row["original_paragraph_indices"]}',
                  f'来源：{row["origin"]}', f'署名：{row["attribution"]}。许可：{row["license"]}（{row["license_url"]}）',
                  f'原文 SHA256：{row["raw_payload_sha256"]}', '', row['text'], '', '审核：待定']
    (D / 'review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    manifest = dict(kind='non_fineweb2_licensed_author_blog_review_r7', status='pending_human_review',
                    source_partition_sha256=checksum(D / 'source-partition.json'),
                    acquisition_plan_sha256=checksum(CYCLE / 'acquisition-plan-r1.json'),
                    parent_manifest_sha256=checksum(PARENT / 'manifest.json'), tokenizer_sha256=tok_sha,
                    source_policy_sha256=checksum(RAW / 'source-policy.json'), historical_review_files=len(reviewed),
                    parent_near_duplicate_probe_rows=len(needles), parent_near_duplicate_matches=len(possible_parent),
                    historical_near_duplicate_matches=len(history_near), rejects=dict(rejects),
                    candidate_train_rows=sum(r['split'] == 'train' for r in selected),
                    candidate_train_tokens=sum(r['tokens_with_eos'] for r in selected if r['split'] == 'train'),
                    candidate_validation_rows=sum(r['split'] == 'validation' for r in selected),
                    candidate_validation_tokens=sum(r['tokens_with_eos'] for r in selected if r['split'] == 'validation'),
                    rows=len(selected), excluded=['fineweb2'], training_started=False, sft_started=False,
                    recorded_utc=datetime.now(timezone.utc).isoformat(),
                    near_duplicate_scope='Parent and historical sources checked using exact/prefix/sentence fingerprints plus normalized CJK 5-gram containment at 78%; 8-CJK probes accelerate parent scan. This does not prove semantic originality.')
    manifest['files'] = {name: checksum(D / name) for name in ('review.json', 'review.md', 'screen.py', 'source-partition.json', 'pre-review-quality-audit.json', 'overlap-probes.txt')}
    save(D / 'manifest.json', manifest)
    print(json.dumps({'historical_review_files': len(reviewed), 'selected': manifest['rows'],
                      'train_rows': manifest['candidate_train_rows'], 'train_tokens': manifest['candidate_train_tokens'],
                      'validation_rows': manifest['candidate_validation_rows'], 'validation_tokens': manifest['candidate_validation_tokens'],
                      'rejects': dict(rejects)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
