"""Record the user's approval of the 62 displayed R1 rows without editing R1 inputs."""
import hashlib
import json
from pathlib import Path


CYCLE = Path(__file__).resolve().parents[1]
R1 = CYCLE / 'non-fineweb2-bulk-intake-r1'
KEY = R1.name
DATE = '2026-11-20'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_json(path, value):
    temp = path.with_name(path.name + '.non-fineweb2-r2.tmp')
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    temp.replace(path)


def main():
    manifest = load(R1/'manifest.json')
    review = load(R1/'review.json')
    for name, digest in manifest['files'].items():
        assert sha(R1/name) == digest
    assert manifest['status'] == review['status'] == 'pending_human_review'
    rows = review['rows']
    assert len(rows) == len({r['id'] for r in rows}) == 62
    assert all(not r['training_eligible'] and not r['validation_admitted'] and r['human_decision'] is None for r in rows)
    assert all(r['source'] in {'globalvoices_zh', 'zhwikiversity', 'zhwikisource'} and
               (r['source']=='globalvoices_zh') == (r['split']=='validation') for r in rows)
    train = [r for r in rows if r['split']=='train']
    validation = [r for r in rows if r['split']=='validation']
    counts = dict(reviewed_rows=len(rows), approved_train_rows=len(train),
                  approved_train_tokens=sum(r['tokens_with_eos'] for r in train),
                  approved_validation_rows=len(validation),
                  approved_validation_tokens=sum(r['tokens_with_eos'] for r in validation))
    assert tuple(counts.values()) == (62, 28, 5289, 34, 6326)
    approval = dict(status='human_review_approved_sample_only', decision='pass', date=DATE,
                    reviewer_record=f'user-confirmed-session-{DATE}',
                    user_confirmation='审核通过，继续扩充数据，记得去重',
                    scope='Only the 62 displayed rows; no corpus freeze, training or SFT authorization.',
                    original_review_sha256=sha(R1/'review.json'), original_manifest_sha256=sha(R1/'manifest.json'),
                    **counts, rows=[dict(id=r['id'], split=r['split'], decision='pass') for r in rows])
    ap = R1/'review-approval.json'
    if ap.exists():
        assert load(ap) == approval
    else:
        replace_json(ap, approval)
    rel = ap.relative_to(CYCLE.parents[3]).as_posix()
    current = dict(path=rel, sha256=sha(ap), status=approval['status'], **counts,
                   scope='Only user-approved sample rows; candidate pool and training remain blocked.')
    rp = CYCLE/'corpus-readiness-r1.json'
    readiness = load(rp)
    approved = readiness['sample_reviewed_pilots']
    totals = readiness['approved_candidate_pilots']
    if KEY not in approved:
        for field, source in (('reviewed_rows','reviewed_rows'), ('approved_unique_train_rows','approved_train_rows'),
                              ('approved_unique_train_tokens','approved_train_tokens'),
                              ('approved_unique_validation_rows','approved_validation_rows'),
                              ('approved_unique_validation_tokens','approved_validation_tokens')):
            totals[field] += counts[source]
        approved[KEY] = current
    else:
        assert approved[KEY] == current
    goals = readiness['admission_targets']
    totals['combined_train_tokens_including_current_snapshot'] = readiness['current_snapshot']['training_tokens'] + totals['approved_unique_train_tokens']
    totals['remaining_unique_train_tokens_to_minimum'] = max(0, goals['unique_train_tokens_minimum']-totals['combined_train_tokens_including_current_snapshot'])
    totals['remaining_independent_validation_tokens_to_minimum'] = max(0, goals['independent_validation_tokens_minimum']-totals['approved_unique_validation_tokens'])
    readiness.update(updated_date=DATE, decision='Approved 62 R1 non-FineWeb2 sample rows. Continue licensed expansion; volume, diversity and mixture gates remain unmet.')
    assert not readiness['training_ready']
    replace_json(rp, readiness)
    cp = CYCLE/'cycle-status-r1.json'
    status = load(cp)
    assert not status['foundation_training_allowed'] and not status['sft_allowed']
    status['candidate_pilots'][KEY] = current
    status['artifacts'][KEY+'-approval'] = dict(path=rel, sha256=sha(ap))
    status.update(updated_date=DATE, status='new_non_fineweb2_sample_approved_corpus_not_ready',
                  next_action='Expand licensed Chinese prose in fixed partitions, with full, paragraph and near-duplicate screening before a fresh review. FineWeb2 remains excluded from new collection.')
    replace_json(cp, status)
    print(counts)


if __name__ == '__main__':
    main()
