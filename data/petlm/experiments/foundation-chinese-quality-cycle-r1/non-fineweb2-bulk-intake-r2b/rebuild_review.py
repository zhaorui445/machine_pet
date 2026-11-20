"""Replace R2 with one passage per previously unseen source document."""
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, unquote, urlsplit


D = Path(__file__).resolve().parent
CYCLE = D.parent
R1 = CYCLE/'non-fineweb2-bulk-intake-r1'
R2 = CYCLE/'non-fineweb2-bulk-intake-r2'
spec = importlib.util.spec_from_file_location('petlm_intake_screen', R1/'screen.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(R2))
from approve_r1 import replace_json

URL = re.compile(r'https?://[^\s<>)）]+')
WIKI = re.compile(r'^zh(?:-\w+)?\.(?:m\.)?(wikibooks|wikivoyage|wikiversity|wikisource)\.org$')


def keyset(url):
    u = urlsplit(url)
    host = u.hostname or ''
    gv_or_cook = base.provenance_key(url)
    if gv_or_cook:
        return {gv_or_cook}
    match = WIKI.fullmatch(host)
    if not match:
        return set()
    source = match[1]
    query = parse_qs(u.query)
    title = query.get('title', [''])[0]
    if '/wiki/' in u.path:
        title = unquote(u.path.split('/wiki/',1)[1])
    title = base.compact(base.normalize(title.replace('_',' '))).casefold()
    oldid = query.get('oldid', [''])[0]
    keys = {f'{source}:title:{title}'} if title else set()
    if oldid.isdigit():
        keys.add(f'{source}:revision:{oldid}')
    return keys


def doc_keys(row):
    return set().union(*(keyset(url) for name in ('origin','url','source_url','history_url')
                         if isinstance(url := row.get(name),str)))


def historical_sources():
    base.D = R2  # Exclude the withdrawn input batch, whose rows are being filtered here.
    paths, _, _, _ = base.history()
    lookup = defaultdict(set)
    for p in sorted(paths):
        if p.suffix=='.md':
            keys = set().union(*(keyset(url.rstrip('，。<>,;')) for url in URL.findall(p.read_text(encoding='utf-8-sig'))))
        else:
            items = base.rows(p) if p.suffix=='.jsonl' else (base.load(p),)
            keys = set().union(*(doc_keys(r) for item in items for r in base.texts(item)))
        for k in keys:
            lookup[k].add(p.relative_to(base.DATA/'experiments').as_posix())
    assert R1/'review.json' in paths
    return paths, lookup


def main():
    source = base.load(R2/'manifest.json')
    assert source['status']=='pending_human_review' and not source['training_started'] and not source['sft_started']
    for name, sha in source['files'].items():
        assert base.checksum(R2/name)==sha
    assert base.checksum(R2/'source-partition.json')==source['source_partition_sha256']
    partition = base.load(R2/'source-partition.json')
    assert partition['excluded']==['fineweb2'] and not any(partition[k] for k in ('admitted','training_allowed','sft_allowed'))
    assert base.checksum(CYCLE/'acquisition-plan-r1.json')==source['acquisition_plan_sha256']
    original = base.load(R2/'review.json')['rows']
    prior = base.load(R1/'review.json')['rows']
    assert len(original)==84 and len(prior)==62
    assert all(r['human_decision'] is None and not r['training_eligible'] and not r['validation_admitted'] for r in original)
    prior_keys = set().union(*(doc_keys(r) for r in prior))
    paths, historical = historical_sources()
    assert prior_keys <= historical.keys()
    used = set()
    drops, kept = [], []
    for i,r in enumerate(original,1):
        keys = doc_keys(r)
        hits = keys & historical.keys()
        if not keys:
            why = 'unverified_document_identity'
        elif keys & prior_keys:
            why = 'same_original_as_r1'
        elif hits:
            why = 'same_original_as_earlier_review'
        elif keys & used:
            why = 'same_original_within_replacement'
        else:
            used.update(keys)
            kept.append(r)
            continue
        drops.append(dict(original_row=i,id=r['id'],title=r['title'],origin=r['origin'],split=r['split'],
                          reason=why, matched_review_files=sorted(set().union(*(historical[k] for k in hits)))))
    assert len(kept)+len(drops)==84 and all(r['split']=='train' for r in kept)
    assert kept and all(partition['sources'][r['source']]['role']==r['split'] for r in kept)
    assert not used & historical.keys()
    groups = defaultdict(list)
    for r in kept:
        groups[(r['source'],r['topic'])].append(r)
    by_reason = dict(Counter(d['reason'] for d in drops))
    tokens = sum(r['tokens_with_eos'] for r in kept)
    lines = ['# 中文正文审核 R2b（按原文去重）','',
             '原 R2 批次已撤回。每篇首次送审的原文只留一条；同一文章换段落或换修订也不会再次送审。',
             '本表只有训练候选。独立验证暂不补入旧文章。FineWeb2 继续排除。', '',
             '| 来源 | 题材 | 待审核条数 | Tokens（含结束符） |','| --- | --- | ---: | ---: |']
    for (src,topic),rs in groups.items():
        lines.append(f'| {src} | {topic} | {len(rs)} | {sum(r["tokens_with_eos"] for r in rs)} |')
    for i,r in enumerate(kept,1):
        lines += ['',f'## {i}. {r["title"]}','',f'来源：{r["origin"]}',
                  f'署名：{r["attribution"]}。权利依据：{r["license"]}（{r["license_url"]}）','',r['text'],'','审核：待定']
    with (D/'review.md').open('x', encoding='utf-8', newline='\n') as out:
        out.write('\n'.join(lines)+'\n')
    base.save(D/'review.json',dict(status='pending_human_review',rows=kept))
    base.save(D/'document-dedup-audit.json',dict(previous_review_files_checked=len(paths),
              original_rows=84, replacement_rows=len(kept), dropped_by_reason=by_reason,
              dropped_rows=drops, previously_reviewed_source_matches_remaining=0))
    manifest = dict(status='pending_human_review', supersedes=R2.name, original_manifest_sha256=base.checksum(R2/'manifest.json'),
                    inherited_source_partition_sha256=source['source_partition_sha256'],
                    acquisition_plan_sha256=source['acquisition_plan_sha256'], parent_manifest_sha256=source['parent_manifest_sha256'],
                    candidate_train_rows=len(kept), candidate_train_tokens=tokens, candidate_validation_rows=0,
                    human_review_completed=False, approved_tokens=0, training_started=False, sft_started=False,
                    validation_note='The 24 R2 validation passages all match earlier source articles. No substitute is counted.',
                    files={name:base.checksum(D/name) for name in ('review.md','review.json','document-dedup-audit.json')})
    base.save(D/'manifest.json',manifest)
    register(manifest)
    print(json.dumps(dict(kept=len(kept), trained=0, train_tokens=tokens, dropped_by_reason=by_reason,
                         historical_review_files=len(paths))))


def register(manifest):
    rel = D.relative_to(CYCLE.parents[3]).as_posix()
    old_rel = R2.relative_to(CYCLE.parents[3]).as_posix()
    withdrawal = dict(status='withdrawn_document_overlap', scope='All 84 original R2 rows withdrawn from review; zero approved.',
                      original_manifest_sha256=manifest['original_manifest_sha256'],
                      replacement=rel+'/manifest.json')
    base.save(R2/'withdrawal.json',withdrawal)
    entry = dict(path=rel+'/manifest.json',review_path=rel+'/review.md',status='pending_human_review',
                 rows=manifest['candidate_train_rows'],candidate_train_rows=manifest['candidate_train_rows'],
                 candidate_train_tokens=manifest['candidate_train_tokens'],candidate_validation_rows=0,
                 approved_tokens=0,training_started=False)
    read_path = CYCLE/'corpus-readiness-r1.json'
    readiness = base.load(read_path)
    assert not readiness['training_ready']
    pending = readiness['pending_candidate_pilots']
    pending[R2.name].update(status=withdrawal['status'], review_path=None, active_review_rows=0, superseded_by=entry['review_path'])
    pending[D.name] = entry
    replace_json(read_path,readiness)
    status_path = CYCLE/'cycle-status-r1.json'
    status = base.load(status_path)
    assert not status['foundation_training_allowed'] and not status['sft_allowed']
    status['candidate_pilots'][R2.name] = pending[R2.name]
    status['candidate_pilots'][D.name] = entry
    for key,path in ((D.name,rel+'/manifest.json'), (R2.name+'-withdrawal',old_rel+'/withdrawal.json')):
        full = CYCLE.parents[3]/path
        status['artifacts'][key] = dict(path=path,sha256=base.checksum(full))
    status['review_dedup_policy'] = dict(document_identity='Exclude previously displayed canonical original articles or chapter pages, including alternate revisions, before text checks. Keep at most one passage per original document per review batch.',
                                         text='Also check full text, paragraphs, and near duplicates against parent, history, and in-batch rows.',
                                         validation='Only unseen articles from fixed validation-only partitions may expand validation reviews. Zero if no eligible new article.',
                                         excluded=['fineweb2'])
    status['next_action'] = 'Review replacement R2b only. Find unseen, licensed validation originals in the fixed validation partition. Training remains blocked.'
    replace_json(status_path,status)


if __name__=='__main__':
    main()
