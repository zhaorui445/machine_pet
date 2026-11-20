"""Record R3 approval and screen R4 from fixed local sources."""
from pathlib import Path
import importlib.util
import re
import sqlite3
from urllib.parse import unquote, urlsplit

D = Path(__file__).resolve().parent
CYCLE = D.parent
R3 = CYCLE/'non-fineweb2-bulk-intake-r3'
GIT = CYCLE/'progit-zh-validation-r1c'
spec = importlib.util.spec_from_file_location('non_fineweb_r3', R3/'expand.py')
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)
base, pipeline = r3.base, r3.pipeline
load, save, sha = base.load, base.save, base.checksum


def register(entry=None):
    m = r3.previous.verify(R3)
    rows = load(R3/'review.json')['rows']
    assert len(rows)==len({r['id'] for r in rows})==35
    assert all(not r['training_eligible'] and not r['validation_admitted'] and r['human_decision'] is None for r in rows)
    counts = dict(reviewed_rows=len(rows))
    for role in ('train','validation'):
        selected = [r for r in rows if r['split']==role]
        counts[f'approved_{role}_rows'] = len(selected)
        counts[f'approved_{role}_tokens'] = sum(r['tokens_with_eos'] for r in selected)
    assert tuple(counts.values())==(35,25,4328,10,2420)
    status_name = 'human_review_approved_sample_only'
    record = dict(status=status_name, date='2026-11-20', decision='pass', reviewer_record='user-confirmed-session-2026-11-20',
                  user_confirmation='审核通过，继续扩充并去重', original_manifest_sha256=sha(R3/'manifest.json'),
                  original_review_sha256=m['files']['review.json'], **counts,
                  rows=[dict(id=r['id'], split=r['split'], decision='pass') for r in rows])
    ap = R3/'review-approval.json'
    if ap.exists():
        assert load(ap)==record
    else:
        save(ap,record)
    approval = dict(path=r3.rel(ap),sha256=sha(ap),status=status_name,**counts)
    rp,cp = CYCLE/'corpus-readiness-r1.json',CYCLE/'cycle-status-r1.json'
    read,status = load(rp),load(cp)
    assert not read['training_ready'] and not status['foundation_training_allowed'] and not status['sft_allowed']
    reviewed,totals = read['sample_reviewed_pilots'],read['approved_candidate_pilots']
    if R3.name not in reviewed:
        totals['reviewed_rows'] += counts['reviewed_rows']
        for role in ('train','validation'):
            for amount in ('rows','tokens'):
                totals[f'approved_unique_{role}_{amount}'] += counts[f'approved_{role}_{amount}']
        reviewed[R3.name] = approval
    else:
        assert reviewed[R3.name]==approval
    totals['combined_train_tokens_including_current_snapshot'] = read['current_snapshot']['training_tokens']+totals['approved_unique_train_tokens']
    targets = read['admission_targets']
    totals['remaining_unique_train_tokens_to_minimum'] = max(0,targets['unique_train_tokens_minimum']-totals['combined_train_tokens_including_current_snapshot'])
    totals['remaining_independent_validation_tokens_to_minimum'] = max(0,targets['independent_validation_tokens_minimum']-totals['approved_unique_validation_tokens'])
    read['pending_candidate_pilots'].pop(R3.name,None)
    read['decision'] = 'R3 sample approved. Foundation corpus remains below the training admission gates.'
    status['candidate_pilots'][R3.name] = approval
    status['artifacts'][R3.name+'-approval'] = dict(path=approval['path'],sha256=approval['sha256'])
    status['next_action'] = 'Screen R4 under the existing article-level split and paragraph deduplication rules.'
    if entry:
        read['pending_candidate_pilots'][D.name] = entry
        status['candidate_pilots'][D.name] = entry
        status['artifacts'][D.name] = dict(path=entry['path'], sha256=sha(D/'manifest.json'))
        status['next_action'] = 'Review R4 passages only; training remains blocked.'
    r3.replace_json(rp,read)
    r3.replace_json(cp,status)


old_keys = r3.previous.docs.doc_keys


def article_keys(row):
    keys = old_keys(row)
    for key in ('origin','url','source_url','history_url'):
        if isinstance(url := row.get(key),str):
            u = urlsplit(url)
            match = re.match(r'^/progit/progit2-zh/(?:blob|commits)/[^/]+/(.+)$',unquote(u.path),re.I)
            if u.hostname=='github.com' and match:
                keys.add('progit:'+match[1].casefold())
    return keys


def add_validation(part, parent, proposals, sources, rejects, history_texts):
    src = 'progit_zh'
    config = part['sources'][src]
    assert config['role']=='validation' and sha(GIT/'manifest.json')==config['manifest_sha256']
    m = load(GIT/'manifest.json')
    for file, key in (('source-partition-r1.json','source_partition_sha256'),('license-evidence-manifest-r1.json','license_evidence_sha256'),
                      ('source-selection-r1.json','source_selection_sha256')):
        assert sha(GIT/file)==m[key]
    evidence = load(GIT/'license-evidence-manifest-r1.json')
    for name,r in evidence['files'].items():
        assert sha(GIT/'license-evidence'/name)==r['sha256']
    source = load(GIT/'source-partition-r1.json')['source']
    assert source['role']=='validation_only' and m['status']=='sample_review_approved_unreviewed_candidates_excluded'
    provenance = base.PARENT/'provenance.jsonl'
    assert sha(provenance)==parent['files'][provenance.name]
    for r in base.rows(provenance):
        u = (r.get('origin') or r.get('url') or '').casefold()
        if 'progit/progit2-zh' in u or re.search(r'git-scm\.com/book/zh',u):
            rejects[f'{src}/source_already_in_parent'] += 1
            return
    hist_fp = set().union(*(base.fingerprints(base.normalize(t)) for t in history_texts))
    log = sources[src] = dict(repo=source['repo'],commit=source['commit'],license=source['license'],scope='validation_only',checked_documents=[])
    sp = base.spm.SentencePieceProcessor(model_proto=(base.TOK/'tokenizer.model').read_bytes())
    match = re.compile(r'^book/(?:01-introduction|03-git-branching|05-distributed-git)/sections/')
    blocked = re.compile(r'^[=+:*>.\[\-]|[\[\]{}<>`]|\+\+\+|https?://|下图|上图|下文|上文|本章|本节|本书|下一节|上一节|以下|示例中|^\s*[$#]',re.M)
    with sqlite3.connect((base.PARENT/'clean-index.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        for doc in m['documents']:
            if not match.search(doc['path']):
                continue
            assert doc['role']=='validation_only' and doc['commit']==source['commit'] and doc['repo']==source['repo']
            f = GIT/doc['raw_path']
            assert sha(f)==doc['raw_sha256'] and f.stat().st_size==doc['bytes']
            log['checked_documents'].append(doc['path'])
            raw = f.read_text(encoding='utf-8-sig')
            title = raw.splitlines()[0].lstrip('= ').strip()
            fenced = False
            for i, block in enumerate(re.split(r'\n\s*\n',raw)):
                boundaries = sum(bool(re.fullmatch(r'-{4,}|\.{4,}|\+{4,}',s.strip())) for s in block.splitlines())
                outside = not fenced
                if boundaries%2:
                    fenced = not fenced
                if boundaries or not outside or blocked.search(block):
                    rejects[f'{src}/format_or_context'] += 1
                    continue
                t = base.normalize(re.sub(r'\s+',' ',block).strip())
                why = base.quality_reason(t)
                if why:
                    rejects[f'{src}/{why}'] += 1
                    continue
                full,prefix = base.canonical_hashes(t)
                if base.fingerprints(t) & hist_fp:
                    rejects[f'{src}/historical_review_exact_prefix_paragraph_or_sentence'] += 1
                    continue
                if db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1',(full,prefix)).fetchone():
                    rejects[f'{src}/parent_index_exact_or_prefix'] += 1
                    continue
                ids = sp.encode(t)
                if sp.unk_id() in ids or base.compact(sp.decode(ids))!=base.compact(t):
                    rejects[f'{src}/tokenizer_roundtrip'] += 1
                    continue
                proposals.append(dict(id=full,source=src,topic='版本管理基础',split='validation',text=t,title=title,
                    origin=doc['origin'],history_url=doc['history_url'],attribution=source['attribution'],license=source['license'],
                    license_url=source['license_url'],raw_payload_sha256=doc['raw_sha256'],paragraph_index=i,
                    tokens_with_eos=len(ids)+1,human_decision=None,training_eligible=False,validation_admitted=False))


def with_validation(p,parent,tokenizer,proposals,sources,rejects,history,files):
    add_validation(p,parent,proposals,sources,rejects,history)
    assert R3/'review.json' in files
    r3.screen_additions(p,parent,tokenizer,proposals,sources,rejects,history,files)


if __name__=='__main__':
    register()
    print('R3 approval recorded for 35 rows.',flush=True)
    part = load(R3/'source-partition.json')
    part['sources']['progit_zh'] = dict(role='validation',manifest_sha256=sha(GIT/'manifest.json'),
                                      scope='Previously pinned noncommercial validation-only Chinese prose; no new retrieval.')
    part['r4_scope'] = 'FineWeb2 excluded. Reuse only verified local snapshots and preserve preexisting publisher and article roles.'
    part['inherited_partition_sha256'] = sha(R3/'source-partition.json')
    part.pop('r3_scope',None)
    target = D/'source-partition.json'
    if target.exists():
        assert load(target)==part
    else:
        save(target,part)
    r3.D = pipeline.D = base.D = D
    r3.previous.docs.doc_keys = article_keys
    r3.native_screen = base.screen
    base.screen = with_validation
    pipeline.main()
    entry = r3.publish_review()
    path = D/'review.md'
    path.write_text(path.read_text(encoding='utf-8').replace('# R3 中文正文审核','# R4 中文正文审核',1),encoding='utf-8',newline='\n')
    m = load(D/'manifest.json')
    m['files']['review.md'] = sha(path)
    assert all(sha(D/name)==digest for name,digest in m['files'].items())
    r3.replace_json(D/'manifest.json',m)
    register(entry)
    print(entry,flush=True)
