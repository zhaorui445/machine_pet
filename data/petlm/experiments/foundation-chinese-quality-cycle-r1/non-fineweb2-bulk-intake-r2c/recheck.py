"""Revalidate the screened R2 passages under the user's paragraph reuse policy."""
from collections import defaultdict
import importlib.util
from pathlib import Path
import re


D = Path(__file__).resolve().parent
CYCLE = D.parent
R2B = CYCLE/'non-fineweb2-bulk-intake-r2b'
spec = importlib.util.spec_from_file_location('intake_document_keys', R2B/'rebuild_review.py')
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)
R1, R2, base = docs.R1, docs.R2, docs.base


def verify(directory):
    manifest = base.load(directory/'manifest.json')
    for name, sha in manifest['files'].items():
        assert base.checksum(directory/name) == sha
    return manifest


def passage_key(row):
    raw = row.get('raw_page_text_sha256') or row.get('raw_payload_sha256')
    indexes = row.get('original_paragraph_indices')
    if indexes is None:
        indexes = [row['paragraph_index']] if 'paragraph_index' in row else []
    return {(key, raw, i) for key in docs.doc_keys(row) for i in indexes} if raw else set()


def sentence_keys(text):
    sentences = [base.normalized(t) for t in re.split(r'[。！？!?\n]+', text)]
    return {s for s in sentences if len(s)>=40}


def main():
    verify(R1)
    m = verify(R2)
    verify(R2B)
    assert m['status']=='pending_human_review' and m['reviewed_rows']==0
    assert not m['training_started'] and not m['sft_started']
    assert base.checksum(R2/'source-partition.json')==m['source_partition_sha256']
    assert base.checksum(CYCLE/'acquisition-plan-r1.json')==m['acquisition_plan_sha256']
    assert base.checksum(base.PARENT/'manifest.json')==m['parent_manifest_sha256']
    part = base.load(R2/'source-partition.json')
    assert part['excluded']==['fineweb2'] and not any(part[k] for k in ('admitted','training_allowed','sft_allowed'))
    prior = base.load(R1/'review.json')['rows']
    candidates = base.load(R2/'review.json')['rows']
    assert len(prior)==62 and len(candidates)==84
    seen_fp, seen_passages, seen_sentences = set(), set(), set()
    seen_text, article_roles = [], {}
    prior_articles = set().union(*(docs.doc_keys(r) for r in prior))
    kept, rejected = [], []
    for is_prior, batch in ((True, prior), (False, candidates)):
        for r in batch:
            keys = docs.doc_keys(r)
            fp, passage = base.fingerprints(r['text']), passage_key(r)
            text, sentences = base.normalized(r['text']), sentence_keys(r['text'])
            roles = {article_roles[key] for key in keys if key in article_roles}
            why = None
            if not is_prior:
                assert r['human_decision'] is None and not r['training_eligible'] and not r['validation_admitted']
                assert part['sources'][r['source']]['role']==r['split']
                why = ('unverified_origin' if not keys else
                       'article_in_other_split' if roles - {r['split']} else
                       'same_source_paragraph' if seen_passages & passage else
                       'same_text_or_sentence' if seen_fp & fp or seen_sentences & sentences else
                       'near_duplicate_or_containment' if any(t in text or text in t or base.similar(t,text) for t in seen_text) else None)
                if why:
                    rejected.append(dict(id=r['id'],title=r['title'],reason=why))
                    continue
                kept.append(dict(r, source_article_seen_in_r1=bool(keys & prior_articles)))
            assert not roles - {r['split']}
            article_roles.update({key:r['split'] for key in keys})
            seen_fp.update(fp)
            seen_passages.update(passage)
            seen_sentences.update(sentences)
            seen_text.append(text)
    assert len(kept)+len(rejected)==84 and kept
    assert len({r['id'] for r in kept})==len(kept)
    train = [r for r in kept if r['split']=='train']
    val = [r for r in kept if r['split']=='validation']
    train_keys = set().union(*(docs.doc_keys(r) for r in train))
    val_keys = set().union(*(docs.doc_keys(r) for r in val))
    assert not train_keys & val_keys
    groups = defaultdict(list)
    for r in kept:
        groups[(r['source'],r['split'],r['topic'])].append(r)
    lines = ['# 中文正文审核 R2c','',
             '同一篇原文可以采用不同段落。已检查段落编号、正文、长句及近似重复；同一篇文章始终归同一组。',
             '「上批用过原文」表示文章相同，正文是不同段落；这类条目只增加段落量。FineWeb2 已排除。','',
             '| 来源 | 用途 | 题材 | 待审核条数 | Tokens（含结束符） |','| --- | --- | --- | ---: | ---: |']
    for (src,split,topic),rs in groups.items():
        role = '训练候选' if split=='train' else '独立验证候选'
        lines.append(f'| {src} | {role} | {topic} | {len(rs)} | {sum(r["tokens_with_eos"] for r in rs)} |')
    for i,r in enumerate(kept,1):
        role = '训练候选' if r['split']=='train' else '独立验证候选'
        coverage = '上批用过原文，本条为不同段落' if r['source_article_seen_in_r1'] else '原文未出现在 R1 审核表'
        index = r.get('original_paragraph_indices') or [r.get('paragraph_index','未提供')]
        lines += ['',f'## {i}. {r["title"]}','',f'用途：{role}；{coverage}；原文段落编号：{", ".join(map(str,index))}',
                  f'来源：{r["origin"]}', f'署名：{r["attribution"]}。权利依据：{r["license"]}（{r["license_url"]}）',
                  '',r['text'],'','审核：待定']
    with (D/'review.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(lines)+'\n')
    base.save(D/'review.json',dict(status='pending_human_review',rows=kept))
    base.save(D/'recheck-audit.json',dict(rechecked_prior_rows=len(prior), input_rows=len(candidates),
              previously_screened_history_files=m['historical_review_files'], rejected=rejected,
              remaining_cross_split_article_keys=len(train_keys & val_keys),
              reused_r1_article_train_rows=sum(r['source_article_seen_in_r1'] for r in train),
              reused_r1_article_validation_rows=sum(r['source_article_seen_in_r1'] for r in val),
              new_validation_articles=len({base.provenance_key(r['origin']) for r in val if not r['source_article_seen_in_r1']})))
    rel = D.relative_to(CYCLE.parents[3]).as_posix()
    entry = dict(path=rel+'/manifest.json',review_path=rel+'/review.md',status='pending_human_review',rows=len(kept),
                 candidate_train_rows=len(train),candidate_train_tokens=sum(r['tokens_with_eos'] for r in train),
                 candidate_validation_rows=len(val),candidate_validation_tokens=sum(r['tokens_with_eos'] for r in val),
                 approved_tokens=0,training_started=False)
    manifest = dict(entry, supersedes=[R2.name,R2B.name], excluded=['fineweb2'], sft_started=False,
                    original_screened_manifest_sha256=base.checksum(R2/'manifest.json'),
                    inherited_source_partition_sha256=m['source_partition_sha256'], acquisition_plan_sha256=m['acquisition_plan_sha256'],
                    files={name:base.checksum(D/name) for name in ('review.md','review.json','recheck-audit.json')})
    base.save(D/'manifest.json',manifest)
    rp, cp = CYCLE/'corpus-readiness-r1.json', CYCLE/'cycle-status-r1.json'
    readiness, status = base.load(rp), base.load(cp)
    assert not readiness['training_ready'] and not status['foundation_training_allowed'] and not status['sft_allowed']
    for registry in (readiness['pending_candidate_pilots'],status['candidate_pilots']):
        for previous in (R2.name,R2B.name):
            registry[previous].update(status='superseded_by_r2c',review_path=None,active_review_rows=0,
                                      superseded_by=entry['review_path'])
        registry[D.name] = entry
    status['artifacts'][D.name] = dict(path=entry['path'],sha256=base.checksum(D/'manifest.json'))
    status['review_dedup_policy'] = dict(document_identity='Different passages from the same original are allowed; one canonical article may belong to only one split.',
        text='Exclude repeated source paragraphs, exact text, long-sentence overlap and near duplicates against parent, historical review and the current batch.',
        validation='Previously used validation-only articles may contribute new passages; count them separately from new articles and publishers.',excluded=['fineweb2'])
    status['next_action'] = 'Review R2c only. Continue licensed intake under the paragraph reuse and source-partition policy; do not train.'
    docs.replace_json(rp,readiness)
    docs.replace_json(cp,status)
    print(entry)
    print(base.load(D/'recheck-audit.json'))


if __name__=='__main__':
    main()
