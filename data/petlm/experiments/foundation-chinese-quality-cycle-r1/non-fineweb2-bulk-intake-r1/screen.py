"""Build a bounded review queue from the previously fixed non-FineWeb2 sources."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import mwparserfromhell
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from petlm.blog_prose import Document, plain, paragraph_runs, quality_reason, VOID
from petlm.foundation_diagnostics import compact
from petlm.licensed_prose_pilot import contiguous_prose, quality_reason as recipe_quality
from petlm.pretrain_increment import canonical_hashes
from petlm.run import checksum
from petlm.wikimedia_prose import fingerprints, normalize, paragraphs, reason

D = Path(__file__).resolve().parent
DATA = D.parents[2]
CYCLE = D.parent
RAW = DATA / 'raw'
PARENT = DATA / 'prepared-chinese-v26r1-strict'
TOK = DATA / 'tokens-chinese-v26r1-strict-24k'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(obj, out, ensure_ascii=False, indent=2)
        out.write('\n')


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            yield json.loads(line)


def classes(node):
    return set((node.attrs.get('class') or '').split())


def provenance_key(url):
    parsed = urlsplit(url)
    if parsed.hostname in {'zh.globalvoices.org', 'zhs.globalvoices.org', 'zht.globalvoices.org'}:
        m = re.search(r'/(\d+)/?$', parsed.path)
        return 'gv:' + m[1] if m else ''
    if parsed.hostname == 'github.com' and parsed.path.lower().startswith('/anduin2017/howtocook/blob/'):
        return 'cook:' + unquote(parsed.path.split('/', 5)[-1])
    return ''


def texts(obj):
    if isinstance(obj, list):
        for el in obj:
            yield from texts(el)
    elif isinstance(obj, dict):
        if isinstance(obj.get('text'), str):
            yield obj
        for el in obj.values():
            if isinstance(el, (dict, list)):
                yield from texts(el)


def history():
    root = DATA / 'experiments'
    paths = set()
    for directory in (CYCLE, root / 'generation-repair-v1'):
        for g in ('review*.json', '*review*.jsonl', '*approved*.jsonl', '*review*.md'):
            paths.update(p for p in directory.rglob(g) if D not in p.parents)
    seen, keys, contents = set(), set(), []
    href = re.compile(r'https?://[^\s<>)]+')
    for p in sorted(paths):
        if p.suffix == '.md':
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                for url in href.findall(line):
                    keys.add(provenance_key(url.rstrip('，。<>,;')))
                if len(line.strip()) >= 60 and not line.startswith(('#', '|', '来源', '原文', 'http')):
                    contents.append(line)
            continue
        iterable = rows(p) if p.suffix == '.jsonl' else [load(p)]
        for obj in iterable:
            for r in texts(obj):
                keys.add(provenance_key(str(r.get('origin') or r.get('url') or '')))
                seen.update(fingerprints(normalize(r['text'])))
                contents.append(r['text'])
    for t in contents:
        seen.update(fingerprints(normalize(t)))
    return paths, seen, keys, contents


def ngrams(t, n=5):
    return {t[i:i+n] for i in range(len(t)-n+1)}


def normalized(t):
    return re.sub(r'[^\w]', '', normalize(t)).lower()


def similar(a, b):
    a, b = ngrams(a), ngrams(b)
    if len(a) < 70 or len(b) < 70:
        return False
    overlap = len(a & b)
    return overlap / min(len(a), len(b)) >= .78


def gv_extract(html, post_id):
    nodes = list(Document(html).root.walk())
    posts = [n for n in nodes if {f'post-{post_id}', 'type-post'} <= classes(n)]
    if len(posts) != 1:
        raise ValueError('no_unique_publisher_post')
    post = list(posts[0].walk())
    bodies = [n for n in post if 'entry' in classes(n)]
    if len(bodies) != 1:
        raise ValueError('no_unique_article_body')
    title = next((plain(n) for n in post if n.tag == 'h1'), '')
    if not title:
        title = next((n.attrs.get('content', '') for n in nodes if n.attrs.get('property') == 'og:title'), '')
    footer = [n for n in post if 'postfooter-credits' in classes(n)]
    if len(footer) != 1:
        raise ValueError('no_unique_article_credits')
    meta = list(footer[0].walk())
    if not any('license' in str(n.attrs.get('rel', '')).split() and
               re.match(r'^https?://creativecommons.org/licenses/by/3\.0/', str(n.attrs.get('href', ''))) for n in meta):
        raise ValueError('article_cc_by_3_0_not_confirmed')
    credits = {}
    for n in meta:
        if 'contributor-name' in classes(n):
            label = next((plain(q) for q in n.walk() if 'credit-label' in classes(q)), '')
            people = [plain(q) for q in n.walk() if q.tag == 'a' and 'user-link' in classes(q) and plain(q)]
            if label and people:
                credits[label] = people
    if not any(re.search(r'written|撰|作者', k, re.I) for k in credits):
        raise ValueError('author_missing')
    if not any(re.search(r'translated|譯|译', k, re.I) for k in credits):
        raise ValueError('translator_missing_or_not_confirmed')
    body = bodies[0]
    if re.search(r'授權轉載|授权转载|本文轉載|本文转载|原載於|原载于|刊登於.{0,15}媒體|保留所有权利|版權所有|版权所有', plain(body)):
        raise ValueError('republishing_or_rights_exception')
    blocks, index, section = [], 0, ''

    def visit(n):
        nonlocal index, section
        if n.tag in {'blockquote', 'script', 'style', 'figure', 'ul', 'ol', 'table', 'aside', 'footer'} or classes(n) & {'wp-caption', 'postmeta-container', 'gv-rss-footer', 'sharedaddy', 'post-credits'}:
            index += 1
            return
        if re.fullmatch(r'h[1-6]', n.tag):
            index += 1
            section = plain(n)
            return
        if n.tag == 'p':
            index += 1
            bad = any(x.tag in VOID - {'br', 'wbr'} or x.tag in {'iframe', 'sup', 'code'} for x in n.walk())
            blocks.append(dict(paragraph_index=index, section=section, original_text=plain(n), structural_rejection=bad))
            return
        for q in n.children:
            if not isinstance(q, str):
                visit(q)

    visit(body)
    return title, credits, blocks


def main():
    p = load(D/'source-partition.json')
    assert 'fineweb2' in p['excluded'] and not p['admitted'] and not p['training_allowed'] and not p['sft_allowed']
    assert checksum(CYCLE/'acquisition-plan-r1.json') == p['acquisition_plan_sha256']
    known = {x['source_group']: x for x in p['sources']}
    st = load(CYCLE/'cycle-status-r1.json')
    assert st['artifacts']['acquisition_plan']['sha256'] == p['acquisition_plan_sha256']
    assert not (D/'manifest.json').exists()
    tokenizer_sha = checksum(TOK/'tokenizer.model')
    assert tokenizer_sha == load(TOK/'manifest.json')['tokenizer_sha256']
    sp = spm.SentencePieceProcessor(model_proto=(TOK/'tokenizer.model').read_bytes())
    reviewed, historic, historic_docs, historic_texts = history()
    print(f'Historical review files: {len(reviewed)}', flush=True)
    parent = load(PARENT/'manifest.json')
    dbmeta = load(PARENT/'clean-index.meta.json')
    assert checksum(PARENT/'manifest.json') == dbmeta['snapshot_manifest_sha256']
    assert checksum(PARENT/'clean-index.sqlite') == dbmeta['index_sha256']
    db = sqlite3.connect((PARENT/'clean-index.sqlite').as_uri()+'?mode=ro', uri=True)
    rejects, sources, proposals = Counter(), {}, []

    def add(source, topic, split, quality, text, **meta):
        text = normalize(text)
        if quality:
            rejects[f'{source}/{quality}'] += 1
            return
        full, prefix = canonical_hashes(text)
        if db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1', (full, prefix)).fetchone():
            rejects[f'{source}/parent_index_exact_or_prefix'] += 1
            return
        if fingerprints(text) & historic:
            rejects[f'{source}/historical_review_exact_prefix_or_sentence'] += 1
            return
        ids = sp.encode(text)
        if sp.unk_id() in ids or compact(sp.decode(ids)) != compact(text):
            rejects[f'{source}/tokenizer_roundtrip'] += 1
            return
        proposals.append(dict(id=full, source=source, topic=topic, split=split, text=text,
                              tokens_with_eos=len(ids)+1, human_decision=None,
                              training_eligible=False, validation_admitted=False, **meta))

    pool = CYCLE/'wikimedia-pilot-r2f'
    pool_meta = load(pool/'manifest.json')
    assert pool_meta['roles']['zhwikiversity']=='train' and known['zhwikiversity']['role']=='train_only'
    assert pool_meta['parent_manifest_sha256']==checksum(PARENT/'manifest.json')
    assert pool_meta['tokenizer_sha256']==tokenizer_sha
    dump = RAW/'wikimedia-zh-source-r2b'/known['zhwikiversity']['snapshot']
    assert checksum(dump)==pool_meta['archive_receipts'][dump.name]['sha256']
    sources['zhwikiversity'] = dict(snapshot_sha256=checksum(dump), mode='reused_attributed_screened_pool', path=str(pool/'train-candidates.jsonl'))
    edu_topic = re.compile(r'中文|语言|閱讀|阅读|寫作|写作|數學|数学|幾何|几何|概率|地理|植物|物理|數字|数字|天文|自然|科學|科学|教育|學習|学习|音樂|音乐|圖書|图书|水|氣候|气候|文學|文学|[語语]文|文化|生物|心理')
    for r in rows(pool/'train-candidates.jsonl'):
        if r['source']!='zhwikiversity' or r['split']!='train' or not edu_topic.search(r['title']):
            rejects['zhwikiversity/outside_education_slice'] += 1
            continue
        add(r['source'], '教育与自然知识', 'train', reason(r['text']), r['text'], **{k:r[k] for k in ('title','origin','history_url','revision_id','attribution','license','license_url','paragraph_index','raw_page_text_sha256')})

    cook = known['Anduin2017/HowToCook']
    assert cook['role']=='train_only_for_listed_paths'
    policy = load(CYCLE/'licensed-prose-pilot-r1/source-policy.json')['sources']['cook']
    assert cook['revision']==policy['commit'] and cook['license'].startswith(policy['text_license'])
    cookraw = RAW/'licensed-chinese-pilot-r2/cook'
    for f, sha in policy['evidence_sha256'].items():
        assert checksum(cookraw/f)==sha
    sources['HowToCook'] = {'revision':cook['revision'],'verified_paths':[]}
    for path, sha1 in cook['paths'].items():
        if 'cook:'+path in historic_docs:
            rejects['HowToCook/path_already_displayed_or_reserved'] += 1
            continue
        fp = cookraw/'text'/path
        b = fp.read_bytes()
        assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==sha1
        sources['HowToCook']['verified_paths'].append(path)
        chunks = list(contiguous_prose(b.decode('utf-8')))
        if not chunks:
            rejects['HowToCook/no_complete_prose_run'] += 1
        for section, ix, text in chunks:
            add('HowToCook','饮食与生活','train',recipe_quality(text),text,title=fp.stem,
                origin=f'https://github.com/{policy["repository"]}/blob/{cook["revision"]}/{path}',
                attribution=policy['attribution'],license=policy['text_license'],license_url=policy['license_url'],
                revision_id=cook['revision'],blob_sha1=sha1,section=section,paragraph_indices=ix)

    wiki = known['zhwikisource']
    assert wiki['role']=='train_only'
    wlog = sources['zhwikisource'] = {}
    for title, rev in wiki['revisions'].items():
        name = D/f'raw-wikisource/{rev}.json'
        rec = load(name.with_suffix('.receipt.json'))
        assert checksum(name)==rec['sha256'] and name.stat().st_size==rec['bytes']
        pg = load(name)['query']['pages'][0]
        record = pg['revisions'][0]
        raw = record['slots']['main']['content']
        assert pg['title']==title and record['revid']==rev and '|author=魯迅' in raw.replace(' ', '')
        issue = ('outside_fixed_publication_year_range_1919' if rev==10675925 else
                 'fixed_revision_only_transcludes_scan_pages' if '<pages ' in raw else
                 'transcription_rated_65_percent_hold' if 'Textquality|65%' in raw else None)
        wlog[title] = dict(revision=rev, sha256=rec['sha256'], status=issue or 'screened_original_public_domain_prose')
        if issue:
            rejects[f'zhwikisource/{issue}'] += 1
            continue
        assert rev==7903272 and '{{Pd/1996|1936}}' in raw and 'Textquality|100%' in raw
        wlog[title]['rights_basis'] = 'Lu Xun (died 1936), first published 1926; fixed page marks PD/1996, 1936. Only original prose, no editorial notes or dates.'
        parsed = mwparserfromhell.parse(raw)
        for template in list(parsed.filter_templates(recursive=False)):
            if str(template.name).strip().lower() in {'header','gap','pd/1996','right'}:
                parsed.remove(template)
        for ix, _, text, why in paragraphs(str(parsed)):
            add('zhwikisource','文学叙事','train',why or reason(text),text or '',title=title,
                origin=f'https://zh.wikisource.org/w/index.php?oldid={rev}',revision_id=rev,
                attribution='鲁迅（1926 年初刊，收入《朝花夕拾》）',license='Public domain (original text)',
                license_url=wiki['rights_reference'], paragraph_index=ix, raw_payload_sha256=rec['sha256'])

    gv = known['publisher:globalvoices']
    assert gv['role']=='validation_only'
    gp = RAW/'author-blogs-r9'
    gvpolicy = load(gp/'globalvoices-policy.json')
    for f, sha in gvpolicy['evidence_sha256'].items():
        assert checksum(gp/'evidence'/f)==sha
    gvlog = sources['globalvoices_zh'] = {'received_files':0, 'screened_posts':[]}
    parent_docs = set()
    provenance = PARENT/'provenance.jsonl'
    assert checksum(provenance)==parent['files'][provenance.name]
    for row in rows(provenance):
        u = str(row.get('origin') or row.get('url') or '')
        if 'globalvoices.org' in u or 'github.com/Anduin2017/HowToCook' in u:
            parent_docs.add(provenance_key(u))
    for f in sorted(RAW.glob('globalvoices-validation-r2*/articles/*.html')):
        gvlog['received_files'] += 1
        receipt = f.with_suffix('.html.receipt.json')
        if not receipt.exists():
            rejects['globalvoices_zh/no_receipt'] += 1
            continue
        r = load(receipt)
        u = r.get('final_url_without_query',r.get('final_url',r['url']))
        key = provenance_key(u)
        if not key.startswith('gv:') or key=='' or provenance_key(r['url'])!=key:
            rejects['globalvoices_zh/archive_url_not_verified'] += 1
            continue
        if key in historic_docs or key in parent_docs:
            rejects['globalvoices_zh/article_already_in_review_or_parent'] += 1
            continue
        assert checksum(f)==r['sha256'] and f.stat().st_size==r['bytes']
        try:
            title, credits, blocks = gv_extract(f.read_text(encoding='utf-8'), key[3:])
        except ValueError as ex:
            rejects[f'globalvoices_zh/{ex}'] += 1
            continue
        gvlog['screened_posts'].append(dict(url=u,title=title,credits=credits,sha256=r['sha256']))
        for item in paragraph_runs(blocks):
            t = item.pop('text')
            add('globalvoices_zh','文化与社会叙事','validation',quality_reason(t),t,title=title,origin=u,
                attribution='；'.join(f'{k}: {", ".join(v)}' for k,v in credits.items()),
                license=gvpolicy['license'],license_url=gvpolicy['license_url'],raw_payload_sha256=r['sha256'],**item)
    db.close()
    print(f'After rights, quality, and first-pass overlap filters: {Counter(r["source"] for r in proposals)}', flush=True)
    screen(p, parent, tokenizer_sha, proposals, sources, rejects, historic_texts, reviewed)


def screen(p, parent, tokenizer_sha, proposals, sources, rejects, historic_texts, reviewed):
    # Straight substrings discover potential fuzzy matches; confirm them on character shingles.
    needles = defaultdict(set)
    tokenized = {}
    for i, r in enumerate(proposals):
        tokenized[i] = normalized(r['text'])
        for seq in re.findall(r'[\u3400-\u9fff]{8,}', r['text']):
            for j in range(0,len(seq)-7,4):
                needles[seq[j:j+8]].add(i)
    pattern_path = D/'overlap-probes.txt'
    pattern_path.write_text('\n'.join(sorted(needles))+'\n', encoding='utf-8')
    if not needles:
        raise ValueError('No searchable initial candidates')
    allwanted = set().union(*(fingerprints(r['text']) for r in proposals))
    parent_hits, parent_fuzzy, history_fuzzy, confirmed = set(), set(), set(), 0
    splits = [PARENT/f'{s}.jsonl' for s in ('train','validation','test')]
    for f in splits:
        assert checksum(f)==parent['files'][f.name]
        for r in rows(f):
            parent_hits.update(fingerprints(r['text']) & allwanted)
        print(f'Parent {f.stem} exact/paragraph overlap complete', flush=True)

    def matches(text, flagged):
        possible = set()
        for seq in re.findall(r'[\u3400-\u9fff]{8,}',text):
            for j in range(len(seq)-7):
                possible.update(needles.get(seq[j:j+8], ()))
        possible -= flagged
        if possible:
            target = normalized(text)
            for i in possible:
                if similar(target,tokenized[i]):
                    flagged.add(i)

    with subprocess.Popen(['rg','--no-config','-F','-f',str(pattern_path),'--no-filename','--',*map(str,splits)],
                          stdout=subprocess.PIPE,encoding='utf-8') as proc:
        for line in proc.stdout:
            matches(json.loads(line)['text'],parent_fuzzy)
            confirmed += 1
        if proc.wait() not in {0,1}:
            raise RuntimeError('Parent substring scan failed')
    for text in historic_texts:
        matches(normalize(text),history_fuzzy)
    selected, seen, bydoc, totals = [], set(), Counter(), Counter()
    for i in sorted(range(len(proposals)),key=lambda i:(proposals[i]['split']!='validation',proposals[i]['source'],proposals[i]['id'])):
        r = proposals[i]
        f = fingerprints(r['text'])
        why = ('parent_exact_prefix_paragraph_or_sentence' if f & parent_hits else
               'parent_near_duplicate' if i in parent_fuzzy else
               'historical_review_near_duplicate' if i in history_fuzzy else
               'candidate_duplicate' if f & seen or any(similar(t,tokenized[i]) for _,t in selected) else None)
        doc = (r['source'],r['title'])
        if why:
            rejects[f'{r["source"]}/{why}'] += 1
        elif bydoc[doc] >= 2 or totals[r['source']] >= 36:
            rejects[f'{r["source"]}/excluded_by_review_batch_bound'] += 1
        else:
            bydoc[doc] += 1
            totals[r['source']] += 1
            seen.update(f)
            selected.append((r,tokenized[i]))
    manifest = dict(status='pending_human_review', source_partition_sha256=checksum(D/'source-partition.json'),
                    acquisition_plan_sha256=p['acquisition_plan_sha256'], parent_manifest_sha256=checksum(PARENT/'manifest.json'),
                    tokenizer_sha256=tokenizer_sha, sources=sources, historical_review_files=len(reviewed),
                    parent_near_duplicate_probe_hits_checked=confirmed, rejects=dict(rejects),
                    reviewed_rows=0, training_started=False, sft_started=False, counts={}, recorded_utc=datetime.now(timezone.utc).isoformat())
    summary = defaultdict(Counter)
    for r,_ in selected:
        summary[(r['source'],r['split'],r['topic'])]['rows'] += 1
        summary[(r['source'],r['split'],r['topic'])]['tokens_with_eos'] += r['tokens_with_eos']
    manifest['counts'] = [dict(source=s,split=d,topic=t,**c) for (s,d,t),c in summary.items()]
    candidates = [r for r,_ in selected]
    save(D/'review.json',dict(status='pending_human_review',rows=candidates))
    lines = ['# 中文正文审核（FineWeb2 已排除）','','本批仅供人工审核。请检查许可和署名、隐私、语句是否自然完整，以及是否有重复。训练与独立验证来源已在正文获取前固定。','',
             '| 来源 | 用途 | 主题 | 待审核条数 | Tokens（含结束符） |','| --- | --- | --- | ---: | ---: |']
    for c in manifest['counts']:
        role = '独立验证' if c['split']=='validation' else '训练候选'
        lines.append(f"| {c['source']} | {role} | {c['topic']} | {c['rows']} | {c['tokens_with_eos']} |")
    for i, r in enumerate(candidates,1):
        lines += ['',f"## {i}. {r['title']}", '',f"来源：{r['origin']}",
                  f"署名：{r['attribution']}。权利依据：{r['license']}（{r['license_url']}）",'',r['text'],'','审核：待定']
    with (D/'review.md').open('x',encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines)+'\n')
    manifest['files'] = {name:checksum(D/name) for name in ('review.json','review.md','screen.py')}
    manifest['near_duplicate_scope'] = 'Candidate 8-CJK probes; confirmed at 78% containment of normalized character 5-grams. This cannot rule out all paraphrases or unseen historical snapshots.'
    save(D/'manifest.json',manifest)
    print(json.dumps(manifest['counts'],ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
