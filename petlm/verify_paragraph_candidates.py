"""Recompute a candidate snapshot's isolation, token counts and review coverage."""
import argparse
import hashlib
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import sentencepiece as spm

from .run import checksum
from .prose_supplement import save
from .foundation_diagnostics import compact
from .wikimedia_prose import fingerprints, reason
from .wikipedia_paragraph_candidates import page_split, extra_reason, paragraphs, verified_rows, prior_exclusions


def verify_subset_membership(subset, upstream):
    source_rows = {r['id']:r for r in upstream}
    seen = set()
    for row in subset:
        parent = source_rows.get(row['id'])
        if (parent is None or row['id'] in seen
                or any(row.get(k) != parent.get(k) for k in ('text','split','split_group','page_id','revision_id','origin','tokens_with_eos'))):
            raise ValueError('Subset rewrites content, duplicates IDs, or moves partitions')
        seen.add(row['id'])


def verify_inherited_subset(root):
    """A pure subset inherits a successful full scan; check every selected row."""
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    source=Path(manifest['parent_candidate_path'])
    source_manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    audit=json.loads((source/'integrity-verification.json').read_text(encoding='utf-8'))
    if (checksum(source/'manifest.json') != manifest['parent_candidate_manifest_sha256']
            or audit['manifest_sha256'] != checksum(source/'manifest.json') or not audit['checks_passed']):
        raise ValueError('Unverified upstream subset')
    for folder, meta in [(root,manifest),(source,source_manifest)]:
        for name,digest in meta['files'].items():
            if checksum(folder/name)!=digest:
                raise ValueError('Subset or upstream changed')
    counts,tokens,groups=Counter(),Counter(),{}
    eligible_ids=set()
    for split in ('train','validation','test'):
        def read(folder):
            with (folder/(split+'-candidates.jsonl')).open(encoding='utf-8') as stream:
                return [json.loads(line) for line in stream]
        rows,parents=read(root),read(source)
        verify_subset_membership(rows,parents)
        counts[split]=len(rows)
        tokens[split]=sum(r['tokens_with_eos'] for r in rows)
        for row in rows:
            if groups.setdefault(row['page_id'],split)!=split:
                raise ValueError('Cross-split group leak')
            if split!='test':
                eligible_ids.add(row['id'])
    review=json.loads((root/'review.json').read_text(encoding='utf-8'))['rows']
    if len(review)!=len(eligible_ids) or {r['id'] for r in review}!=eligible_ids:
        raise ValueError('Review coverage mismatch')
    if dict(counts)!=manifest['candidate_counts'] or dict(tokens)!=manifest['candidate_tokens_with_eos']:
        raise ValueError('Subset counts mismatch')
    result=dict(audit,kind='verified_inherited_candidate_subset_v1',
        verified_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=checksum(root/'manifest.json'),
        inherited_full_audit_path=str(source/'integrity-verification.json'),
        inherited_full_audit_sha256=checksum(source/'integrity-verification.json'),
        counts=dict(counts),tokens_with_eos=dict(tokens),review_rows=len(review),
        page_groups_by_split=dict(Counter(groups.values())),largest_single_page_token_share=None,
        inheritance_basis='Every selected row keeps exact upstream text, token count, page group and partition; no new rows.')
    save(root/'integrity-verification.json',result)
    print(json.dumps(result,ensure_ascii=False))


def verify(root, base, tokenizer_dir):
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if checksum(root/name) != expected:
            raise ValueError(f'Snapshot changed: {name}')
    parent = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if checksum(base/'manifest.json') != manifest['parent_manifest_sha256']:
        raise ValueError('Parent snapshot changed')
    if checksum(tokenizer_dir/'tokenizer.model') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    tokenizer = spm.SentencePieceProcessor(model_proto=(tokenizer_dir/'tokenizer.model').read_bytes())
    all_hashes, seen_ids, groups, review_ids = set(), set(), {}, set()
    candidates_by_page = {}
    counts, tokens = Counter(), Counter()
    max_page_tokens = Counter()
    for split in ('train', 'validation', 'test'):
        with (root/(split+'-candidates.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['split'] != split or page_split(row['page_id']) != split:
                    raise ValueError('Unstable article split')
                if groups.setdefault(row['page_id'], split) != split:
                    raise ValueError('Article appears across splits')
                if row['id'] in seen_ids or fingerprints(row['text']) & all_hashes:
                    raise ValueError('Duplicate candidate content')
                if row['training_eligible'] or row['human_review_status'] != 'pending':
                    raise ValueError('Unexpected approval or training eligibility')
                if reason(row['text']) or extra_reason(row['text']):
                    raise ValueError('Screening is not idempotent')
                ids = tokenizer.encode(row['text'])
                if (tokenizer.unk_id() in ids or compact(tokenizer.decode(ids)) != compact(row['text'])
                        or len(ids)+1 != row['tokens_with_eos']):
                    raise ValueError('Token count or roundtrip failed')
                seen_ids.add(row['id'])
                candidates_by_page.setdefault(row['page_id'], []).append(row)
                all_hashes.update(fingerprints(row['text']))
                counts[split] += 1
                tokens[split] += len(ids)+1
                max_page_tokens[(split,row['page_id'])] += len(ids)+1
                if split != 'test':
                    review_ids.add(row['id'])
    if dict(counts) != manifest['candidate_counts'] or dict(tokens) != manifest['candidate_tokens_with_eos']:
        raise ValueError('Manifest totals differ from recomputed totals')
    review = json.loads((root/'review.json').read_text(encoding='utf-8'))['rows']
    if len(review) != len(review_ids) or {r['id'] for r in review} != review_ids:
        raise ValueError('Incomplete or contaminated review coverage')
    for source in manifest['sources']:
        path = Path(source['path'])
        if checksum(path) != source['sha256']:
            raise ValueError('Raw source changed')
        for _, page in verified_rows(path, source):
            if str(page['id']) not in candidates_by_page:
                continue
            blocks = {i:(text, why, raw) for i,_,text,why,raw in paragraphs(page['raw_wikitext'])}
            for row in candidates_by_page[str(page['id'])]:
                if row['archive_file'] != source['archive_file']:
                    continue
                text, why, raw = blocks[row['paragraph_index']]
                if (why or text != row['text'] or row['revision_id'] != str(page['revision_id'])
                        or row['raw_paragraph_sha256'] != hashlib.sha256(raw.encode()).hexdigest()
                        or row['raw_page_text_sha256'] != hashlib.sha256(page['raw_wikitext'].encode()).hexdigest()):
                    raise ValueError('Original paragraph reconstruction failed')
    parent_pages, prior_hashes, _ = prior_exclusions(base, parent, [Path(r['path']) for r in manifest['prior_candidate_exclusions']])
    if parent_pages & set(groups) or prior_hashes & all_hashes:
        raise ValueError('Parent page or previous candidate overlap')
    parent_rows = Counter()
    for split in ('train','validation','test'):
        path = base/(split+'.jsonl')
        if checksum(path) != parent['files'][path.name]:
            raise ValueError('Parent data changed')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if fingerprints(json.loads(line)['text']) & all_hashes:
                    raise ValueError('Parent paragraph or sentence overlap')
                parent_rows[split] += 1
        print(f'Verification: parent {split} scanned.', flush=True)
    if dict(parent_rows) != parent['splits']:
        raise ValueError('Parent row count mismatch')
    result = {'kind':'candidate_integrity_verification_v1', 'verified_utc':datetime.now(timezone.utc).isoformat(),
        'manifest_sha256':checksum(manifest_path), 'checks_passed':True,
        'counts':dict(counts), 'tokens_with_eos':dict(tokens), 'review_rows':len(review),
        'page_groups_by_split':dict(Counter(groups.values())),
        'largest_single_page_token_share':{s:max((t for (part,_),t in max_page_tokens.items() if part==s),default=0)/max(tokens[s],1) for s in counts},
        'parent_rows_checked':dict(parent_rows), 'parent_fingerprint_overlap':0,
        'candidate_fingerprint_overlap':0, 'cross_split_page_overlap':0,
        'tokenizer_roundtrip_failures':0, 'human_review_completed':False,
        'original_paragraph_reconstruction_passed':True,
        'prior_candidate_fingerprint_overlap':0, 'parent_page_id_overlap':0,
        'training_ready':False, 'training_started':False,
        'test_text_displayed':False, 'frozen_blind_benchmark_used':False,
        'limitations':['Fingerprint checks are not semantic-near-duplicate detection.',
            'Checks cover the recorded parent snapshot, not every unrecorded ancestral training source.',
            'Mechanical validation does not certify naturalness, truthfulness or rights for third-party quotations.']}
    save(root/'integrity-verification.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--verified-subset', action='store_true')
    parser.add_argument('--base', type=Path, default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    parser.add_argument('--tokenizer-dir', type=Path, default=Path('data/petlm/tokens-chinese-v26r1-strict-24k'))
    args = parser.parse_args()
    if args.verified_subset:
        verify_inherited_subset(args.candidate)
    else:
        verify(args.candidate, args.base, args.tokenizer_dir)
