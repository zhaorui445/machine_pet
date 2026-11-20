"""Expand the licensed intake with verified local archives and a fresh review."""
from collections import Counter
import importlib.util
from pathlib import Path
import re
import sqlite3


D = Path(__file__).resolve().parent
R1 = D.parent/'non-fineweb2-bulk-intake-r1'
spec = importlib.util.spec_from_file_location('petlm_intake_r1', R1/'screen.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.D = D

from petlm.wikimedia_prose_select import classify, origin_group


TOPICS = {
    'learning_and_science': '教育与科学', 'learning_and_memory': '学习与记忆',
    'daily_travel': '日常出行', 'food': '饮食与生活', 'nature': '自然与地理',
}
EDU = re.compile(r'语言|阅读|写作|数学|几何|概率|地理|植物|物理|数字|天文|自然|科学|教育|学习|音乐|图书|气候|文学|语文|生物')
EXCLUDE = re.compile(r'转载|摘自|版权所有|原载于|以下是|图中|所示|下文|上文|住在.{0,8}[路街].{0,3}[0-9]+号')


def verified(path, expected):
    actual = base.checksum(path)
    if actual != expected:
        raise ValueError(f'Hash mismatch: {path}')
    return actual


def main():
    part = base.load(D/'source-partition.json')
    cycle, parent_dir, raw, tok = base.CYCLE, base.PARENT, base.RAW, base.TOK
    assert part['excluded']==['fineweb2'] and not any(part[k] for k in ('admitted','training_allowed','sft_allowed'))
    verified(cycle/'acquisition-plan-r1.json',part['acquisition_plan_sha256'])
    prior = base.load(R1/'review-approval.json')
    assert prior['reviewed_rows']==62 and prior['decision']=='pass'
    verified(R1/'review.json',prior['original_review_sha256'])
    verified(R1/'manifest.json',prior['original_manifest_sha256'])
    verified(R1/'screen.py',base.load(R1/'manifest.json')['files']['screen.py'])
    parent = base.load(parent_dir/'manifest.json')
    parent_sha = base.checksum(parent_dir/'manifest.json')
    idx = base.load(parent_dir/'clean-index.meta.json')
    assert idx['snapshot_manifest_sha256']==parent_sha
    verified(parent_dir/'clean-index.sqlite',idx['index_sha256'])
    tokenizer_sha = verified(tok/'tokenizer.model',base.load(tok/'manifest.json')['tokenizer_sha256'])
    tokenizer = base.spm.SentencePieceProcessor(model_proto=(tok/'tokenizer.model').read_bytes())

    files, history_fp, history_docs, history_texts = base.history()
    assert R1/'review.json' in files
    prior_review = base.load(R1/'review.json')['rows']
    prior_gv = {base.provenance_key(r['origin']) for r in prior_review if r['source']=='globalvoices_zh'}
    # Compare with every displayed paragraph, including previously rejected rows.
    print(f'History files checked: {len(files)}',flush=True)
    provenance = parent_dir/'provenance.jsonl'
    verified(provenance,parent['files'][provenance.name])
    parent_docs = set()
    parent_groups = set()
    for r in base.rows(provenance):
        url = r.get('origin') or r.get('url') or ''
        if 'globalvoices.org' in url:
            parent_docs.add(base.provenance_key(url))
        if 'wikibooks.org' in url or 'wikivoyage.org' in url:
            group = origin_group(url)
            if group:
                parent_groups.add(group)
    db = sqlite3.connect((parent_dir/'clean-index.sqlite').as_uri()+'?mode=ro',uri=True)
    reject, sources, proposals = Counter(), {}, []

    def add(src, topic, text, **meta):
        text = base.normalize(text)
        full, prefix = base.canonical_hashes(text)
        if EXCLUDE.search(text):
            reject[f'{src}/source_exception_or_incomplete_context'] += 1
        elif base.fingerprints(text) & history_fp:
            reject[f'{src}/historical_review_exact_prefix_paragraph_or_sentence'] += 1
        elif db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1',(full,prefix)).fetchone():
            reject[f'{src}/parent_index_exact_or_prefix'] += 1
        else:
            ids = tokenizer.encode(text)
            if tokenizer.unk_id() in ids or base.compact(tokenizer.decode(ids))!=base.compact(text):
                reject[f'{src}/tokenizer_roundtrip'] += 1
                return
            proposals.append(dict(id=full, source=src, topic=topic, split=part['sources'][src]['role'],
                              text=text, tokens_with_eos=len(ids)+1, human_decision=None,
                              training_eligible=False, validation_admitted=False, **meta))

    for pilot, items, raw_dir, receipt_key in (
        ('wikimedia-pilot-r1', ('zhwikibooks','zhwikivoyage'), 'wikimedia-zh-source-r1', 'source_receipts'),
        ('wikimedia-pilot-r2f', ('zhwikiversity',), 'wikimedia-zh-source-r2b', 'archive_receipts'),
    ):
        pool = cycle/pilot
        m = base.load(pool/'manifest.json')
        assert m['parent_manifest_sha256']==parent_sha and m['tokenizer_sha256']==tokenizer_sha
        f = pool/'train-candidates.jsonl'
        held = set()
        for name in ('validation-candidates.jsonl','test-candidates.jsonl'):
            if (pool/name).exists():
                held.update(r['split_group'] for r in base.rows(pool/name))
        for src in items:
            assert part['sources'][src]['role']=='train'
            name = src+'-20261101-pages-articles.xml.bz2'
            digest = verified(raw/raw_dir/name,m[receipt_key][name]['sha256'])
            sources[src] = dict(snapshot=name, sha256=digest, pool_sha256=base.checksum(f), old_role='train')
        for r in base.rows(f):
            src = r['source']
            if src not in items:
                continue
            assert r['split']=='train' and r['license']=='CC-BY-SA-4.0'
            if r['split_group'] in held or r['split_group'] in parent_groups:
                reject[f'{src}/held_out_or_parent_document_group'] += 1
                continue
            if src=='zhwikiversity':
                genre = 'learning_and_science' if EDU.search(r['title']) else None
                why = None if genre else 'outside_education_slice'
            else:
                genre, why = classify(r)
                # Ordinary travel and food prose are preferred over technical tourism advice.
                if not why and src=='zhwikivoyage' and genre not in {'daily_travel','food','nature'}:
                    why = 'outside_daily_travel_scope'
            why = why or base.reason(r['text'])
            if why:
                reject[f'{src}/{why}'] += 1
                continue
            attrs = ('title','origin','history_url','revision_id','attribution','license','license_url','paragraph_index','raw_page_text_sha256')
            add(src, TOPICS[genre], r['text'], split_group=r['split_group'], **{k:r[k] for k in attrs})

    gv = 'globalvoices_zh'
    assert part['sources'][gv]['role']=='validation'
    policy_dir = raw/'author-blogs-r9'
    policy = base.load(policy_dir/'globalvoices-policy.json')
    assert policy['license']=='CC-BY-3.0' and policy['split']=='validation'
    for name, digest in policy['evidence_sha256'].items():
        verified(policy_dir/'evidence'/name,digest)
    sources[gv] = dict(mode='verified_local_publisher_copies', articles=[], source_partition='publisher:globalvoices',
                       note='Unseen passages may come from R1 validation-only articles. Such passages expand text coverage, not publisher or article coverage.')
    seen_docs = set()
    for f in sorted(raw.glob('globalvoices-validation-r2*/articles/*.html')):
        receipt = f.with_suffix('.html.receipt.json')
        if not receipt.exists():
            reject[f'{gv}/no_receipt'] += 1
            continue
        rec = base.load(receipt)
        url = rec.get('final_url_without_query',rec.get('final_url',rec['url']))
        key = base.provenance_key(url)
        if not key.startswith('gv:') or base.provenance_key(rec['url'])!=key:
            reject[f'{gv}/source_url_not_verified'] += 1
            continue
        if key in seen_docs or key in parent_docs or (key in history_docs and key not in prior_gv):
            reject[f'{gv}/duplicate_archive_parent_or_older_review_article'] += 1
            continue
        assert f.stat().st_size==rec['bytes']
        digest = verified(f,rec['sha256'])
        try:
            title, credits, blocks = base.gv_extract(f.read_text(encoding='utf-8'),key[3:])
        except ValueError as ex:
            reject[f'{gv}/{ex}'] += 1
            continue
        seen_docs.add(key)
        sources[gv]['articles'].append(dict(url=url,title=title,sha256=digest,seen_in_r1=key in prior_gv))
        for item in base.paragraph_runs(blocks):
            text = item.pop('text')
            why = base.quality_reason(text)
            if why:
                reject[f'{gv}/{why}'] += 1
                continue
            add(gv,'文化与社会叙事',text,title=title,origin=url,source_article_seen_in_r1=key in prior_gv,
                split_group='publisher:globalvoices',attribution='；'.join(f'{k}: {", ".join(v)}' for k,v in credits.items()),
                license=policy['license'],license_url=policy['license_url'],raw_payload_sha256=digest,**item)
    db.close()
    print(f'Candidates after first pass: {Counter(r["source"] for r in proposals)}',flush=True)
    base.screen(part,parent,tokenizer_sha,proposals,sources,reject,history_texts,files)


if __name__=='__main__':
    main()
