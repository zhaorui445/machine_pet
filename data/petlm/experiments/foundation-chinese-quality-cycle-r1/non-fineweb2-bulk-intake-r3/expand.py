"""Record R2c approval and screen the next batch from the pinned source pools."""
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path


D = Path(__file__).resolve().parent
CYCLE = D.parent
R2 = CYCLE/'non-fineweb2-bulk-intake-r2'
R2C = CYCLE/'non-fineweb2-bulk-intake-r2c'
DATE = '2026-11-20'


def import_file(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = import_file('intake_r2c', R2C/'recheck.py')
pipeline = import_file('intake_r2', R2/'screen.py')
base = pipeline.base
replace_json = previous.docs.replace_json
rel = lambda path: path.relative_to(CYCLE.parents[3]).as_posix()


def approve():
    m = previous.verify(R2C)
    rows = base.load(R2C/'review.json')['rows']
    assert len(rows)==len({r['id'] for r in rows})==84
    train = [r for r in rows if r['split']=='train']
    val = [r for r in rows if r['split']=='validation']
    assert all(r['human_decision'] is None and not r['training_eligible'] and not r['validation_admitted'] for r in rows)
    counts = dict(reviewed_rows=len(rows), approved_train_rows=len(train), approved_train_tokens=sum(r['tokens_with_eos'] for r in train),
                  approved_validation_rows=len(val),approved_validation_tokens=sum(r['tokens_with_eos'] for r in val))
    assert tuple(counts.values())==(84,60,10315,24,3807)
    record = dict(status='human_review_approved_sample_only',date=DATE,decision='pass',
        reviewer_record=f'user-confirmed-session-{DATE}',user_confirmation='审核通过，继续扩充并查重',
        original_manifest_sha256=base.checksum(R2C/'manifest.json'), original_review_sha256=m['files']['review.json'],
        **counts,rows=[dict(id=r['id'],split=r['split'],decision='pass') for r in rows])
    approval = R2C/'review-approval.json'
    if approval.exists():
        assert base.load(approval)==record
    else:
        base.save(approval,record)
    return dict(path=rel(approval),sha256=base.checksum(approval),status=record['status'],**counts)


def update_registries(approval, entry=None):
    rp,cp = CYCLE/'corpus-readiness-r1.json',CYCLE/'cycle-status-r1.json'
    readiness,status = base.load(rp),base.load(cp)
    assert not readiness['training_ready'] and not status['foundation_training_allowed'] and not status['sft_allowed']
    totals = readiness['approved_candidate_pilots']
    approved = readiness['sample_reviewed_pilots']
    if R2C.name not in approved:
        for dst,src in (('reviewed_rows','reviewed_rows'),('approved_unique_train_rows','approved_train_rows'),
                        ('approved_unique_train_tokens','approved_train_tokens'),('approved_unique_validation_rows','approved_validation_rows'),
                        ('approved_unique_validation_tokens','approved_validation_tokens')):
            totals[dst] += approval[src]
        approved[R2C.name] = approval
    else:
        assert approved[R2C.name]==approval
    targets = readiness['admission_targets']
    totals['combined_train_tokens_including_current_snapshot'] = readiness['current_snapshot']['training_tokens']+totals['approved_unique_train_tokens']
    totals['remaining_unique_train_tokens_to_minimum'] = max(0,targets['unique_train_tokens_minimum']-totals['combined_train_tokens_including_current_snapshot'])
    totals['remaining_independent_validation_tokens_to_minimum'] = max(0,targets['independent_validation_tokens_minimum']-totals['approved_unique_validation_tokens'])
    readiness['pending_candidate_pilots'].pop(R2C.name,None)
    readiness['decision'] = 'R2c sample approved; foundation volume, source mixture and validation coverage gates remain unmet.'
    status['candidate_pilots'][R2C.name] = approval
    status['artifacts'][R2C.name+'-approval'] = dict(path=approval['path'],sha256=approval['sha256'])
    status['next_action'] = 'Screen the next batch of licensed prose in the fixed source partitions; FineWeb2 excluded.'
    if entry:
        readiness['pending_candidate_pilots'][D.name] = entry
        status['candidate_pilots'][D.name] = entry
        status['artifacts'][D.name] = dict(path=entry['path'],sha256=base.checksum(D/'manifest.json'))
        status['next_action'] = 'Review R3 passages only. Keep the existing article assignments and do not start training.'
    replace_json(rp,readiness)
    replace_json(cp,status)


def screen_additions(p,parent,tokenizer_sha,proposals,sources,rejects,history_texts,reviewed):
    prior_roles = defaultdict(set)
    prior_passages, seen_sentence = set(), set()
    assert R2C/'review.json' in reviewed
    for f in sorted(reviewed):
        if f.suffix not in {'.json','.jsonl'}:
            continue
        items = base.rows(f) if f.suffix=='.jsonl' else (base.load(f),)
        for item in items:
            for r in base.texts(item):
                if r.get('split') not in {'train','validation','test'}:
                    continue
                for k in previous.docs.doc_keys(r):
                    prior_roles[k].add(r['split'])
                prior_passages.update(previous.passage_key(r))
                seen_sentence.update(previous.sentence_keys(r['text']))
    accepted = []
    for r in proposals:
        keys = previous.docs.doc_keys(r)
        old_roles = set().union(*(prior_roles[k] for k in keys))
        if not keys or old_roles-{r['split']}:
            rejects[f'{r["source"]}/unverified_article_or_historical_cross_split'] += 1
        elif previous.passage_key(r) & prior_passages:
            rejects[f'{r["source"]}/historical_source_paragraph'] += 1
        elif previous.sentence_keys(r['text']) & seen_sentence:
            rejects[f'{r["source"]}/historical_normalized_sentence_overlap'] += 1
        else:
            r['article_seen_in_historical_review'] = bool(old_roles)
            accepted.append(r)
    print(f'After article and source paragraph checks: {Counter(r["source"] for r in accepted)}',flush=True)
    native_screen(p,parent,tokenizer_sha,accepted,sources,rejects,history_texts,reviewed)


def publish_review():
    m = base.load(D/'manifest.json')
    rows = base.load(D/'review.json')['rows']
    train = [r for r in rows if r['split']=='train']
    val = [r for r in rows if r['split']=='validation']
    assert rows and len(rows)==len({r['id'] for r in rows})
    assert all(r['source']!='fineweb2' and not r['training_eligible'] and not r['validation_admitted'] and r['human_decision'] is None for r in rows)
    train_keys = set().union(*(previous.docs.doc_keys(r) for r in train))
    val_keys = set().union(*(previous.docs.doc_keys(r) for r in val))
    assert not train_keys & val_keys
    old = base.load(R2C/'review.json')['rows']+base.load(previous.R1/'review.json')['rows']
    fingerprints, passages, seen = set(), set(), []
    for idx,r in enumerate(old+rows):
        f,p,t = base.fingerprints(r['text']),previous.passage_key(r),base.normalized(r['text'])
        if idx>=len(old):
            assert not (f & fingerprints or p & passages or any(base.similar(s,t) for s in seen))
        fingerprints.update(f)
        passages.update(p)
        seen.append(t)
    lines = ['# R3 中文正文审核','',
        '正文已与父语料、历史审核文本和本批内容查重。同一原文可采用不同段落，但只能属于训练或验证其中一组。FineWeb2 已排除。','',
        '| 来源 | 用途 | 待审核条数 | Tokens（含结束符） |','| --- | --- | ---: | ---: |']
    for c in m['counts']:
        role = '训练候选' if c['split']=='train' else '独立验证候选'
        lines.append(f'| {c["source"]} | {role} | {c["rows"]} | {c["tokens_with_eos"]} |')
    for i,r in enumerate(rows,1):
        role = '训练候选' if r['split']=='train' else '独立验证候选'
        tag = '已审核原文的不同段落' if r['article_seen_in_historical_review'] else '未出现在历史审核记录的原文'
        index = r.get('original_paragraph_indices') or [r.get('paragraph_index','未提供')]
        lines += ['',f'## {i}. {r["title"]}','',f'用途：{role}；{tag}；原文段落编号：{", ".join(map(str,index))}',
                  f'来源：{r["origin"]}', f'署名：{r["attribution"]}。权利依据：{r["license"]}（{r["license_url"]}）',
                  '',r['text'],'','审核：待定']
    review = D/'review.md'
    review.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    m['files']['review.md'] = base.checksum(review)
    m['new_validation_articles'] = len({r['origin'] for r in val if not r['article_seen_in_historical_review']})
    m['reused_validation_articles'] = len({r['origin'] for r in val if r['article_seen_in_historical_review']})
    m['approved_tokens'] = 0
    replace_json(D/'manifest.json',m)
    return dict(path=rel(D/'manifest.json'),review_path=rel(review),status='pending_human_review',rows=len(rows),
        candidate_train_rows=len(train),candidate_train_tokens=sum(r['tokens_with_eos'] for r in train),
        candidate_validation_rows=len(val),candidate_validation_tokens=sum(r['tokens_with_eos'] for r in val),
        new_validation_articles=m['new_validation_articles'],reused_validation_articles=m['reused_validation_articles'],
        approved_tokens=0,training_started=False)


if __name__=='__main__':
    approval = approve()
    update_registries(approval)
    print(f'Recorded review approval for {approval["reviewed_rows"]} R2c rows',flush=True)
    old_partition = R2/'source-partition.json'
    partition = base.load(old_partition)
    partition['r3_scope'] = 'Existing pinned archive pools with inherited roles. Additional unseen passages are allowed within the same article and role.'
    partition['inherited_partition_sha256'] = base.checksum(old_partition)
    if (D/'source-partition.json').exists():
        assert base.load(D/'source-partition.json')==partition
    else:
        base.save(D/'source-partition.json',partition)
    pipeline.D = base.D = D
    native_screen = base.screen
    base.screen = screen_additions
    pipeline.main()
    entry = publish_review()
    update_registries(approval,entry)
    print(entry,flush=True)
