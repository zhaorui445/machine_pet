"""Verify the displayed queue before registering it as pending review."""
from collections import defaultdict
import hashlib

from approve_r1 import load, replace_json, CYCLE, R1
from screen import D, base


def main():
    rows = load(D/'review.json')['rows']
    m = load(D/'manifest.json')
    assert m['status']=='pending_human_review' and not m['training_started'] and not m['sft_started']
    assert len(rows)==len({r['id'] for r in rows})==84
    assert all(r['human_decision'] is None and not r['training_eligible'] and not r['validation_admitted'] for r in rows)
    for name, sha in m['files'].items():
        assert base.checksum(D/name)==sha
    p = load(D/'source-partition.json')
    assert base.checksum(D/'source-partition.json')==m['source_partition_sha256']
    assert all(p['sources'][r['source']]['role']==r['split'] for r in rows)
    assert p['excluded']==['fineweb2']
    counts = defaultdict(lambda: [0,0])
    for r in rows:
        counts[r['split']][0] += 1
        counts[r['split']][1] += r['tokens_with_eos']
    assert dict(counts)=={'train':[60,10315], 'validation':[24,3807]}
    prior = load(R1/'review.json')['rows']
    seen = set().union(*(base.fingerprints(r['text']) for r in prior))
    texts = [base.normalized(r['text']) for r in prior]
    for r in rows:
        fp = base.fingerprints(r['text'])
        normalized = base.normalized(r['text'])
        assert not (seen & fp) and not any(base.similar(normalized,t) for t in texts)
        seen.update(fp)
        texts.append(normalized)
    validation = [r for r in rows if r['split']=='validation']
    docs = {base.provenance_key(r['origin']) for r in validation}
    existing_docs = {base.provenance_key(r['origin']) for r in prior if r['split']=='validation'}
    assert all(k.startswith('gv:') for k in docs) and len(docs)==12 and docs <= existing_docs
    assert all(r['source_article_seen_in_r1'] for r in validation)
    rel = D.relative_to(CYCLE.parents[3]).as_posix()
    entry = dict(path=rel+'/manifest.json', review_path=rel+'/review.md', status='pending_human_review',
                 rows=len(rows), candidate_train_rows=60, candidate_train_tokens=10315,
                 candidate_validation_rows=24, candidate_validation_tokens=3807,
                 validation_source_articles=12, new_validation_source_articles=0, approved_tokens=0,
                 training_started=False,
                 scope='New candidate passages; Global Voices articles are validation only and were already sampled in R1.')
    rp = CYCLE/'corpus-readiness-r1.json'
    readiness = load(rp)
    assert not readiness['training_ready']
    readiness['pending_candidate_pilots'][D.name] = entry
    replace_json(rp,readiness)
    cp = CYCLE/'cycle-status-r1.json'
    status = load(cp)
    assert not status['foundation_training_allowed'] and not status['sft_allowed']
    status['candidate_pilots'][D.name] = entry
    status['artifacts'][D.name] = dict(path=entry['path'], sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest())
    status['next_action'] = 'Review the 84 non-FineWeb2 R2 candidate passages. Continue seeking independently sourced validation from new articles and publishers. Training remains blocked.'
    replace_json(cp,status)
    print('Verified 84 rows, both split totals, frozen inputs and no exact or near overlap with R1 or within R2.')


if __name__=='__main__':
    main()
