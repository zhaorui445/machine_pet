"""Reduce review concentration without rewriting source prose or approving it."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata

from .prose_supplement import save
from .run import checksum


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if c.isalnum())


def similarity(a, b):
    a, b = normalized(a), normalized(b)
    if a == b:
        return 1.0
    if min(len(a), len(b)) < 80:
        return 0.0
    left = {a[i:i+4] for i in range(len(a)-3)}
    right = {b[i:i+4] for i in range(len(b)-3)}
    return len(left & right)/min(len(left), len(right))


def reduce_rows(rows, clusters=None, preferred=(), threshold=.75):
    """Keep held-out priority; source and manually assigned clusters cap at one."""
    clusters = clusters or {}
    preferred = set(preferred)
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated candidate ID')
    order = {r['id']:i for i,r in enumerate(rows)}
    kept, removed, source_seen, cluster_seen = [], [], {}, {}
    for row in sorted(rows, key=lambda r: ({'validation':0, 'train':1}[r['split']],
                      r['id'] not in preferred, order[r['id']])):
        match = next((x for x in kept if similarity(x['text'], row['text']) >= threshold), None)
        reason = 'lexical_near_duplicate' if match else None
        source = row['split_group']
        cluster = clusters.get(row['id'])
        if match is None and source in source_seen:
            match, reason = source_seen[source], 'same_source_article_cap'
        if match is None and cluster and cluster in cluster_seen:
            match, reason = cluster_seen[cluster], 'similar_topic_and_description_cap'
        if match is not None:
            removed.append({'id':row['id'], 'title':row['title'], 'split':row['split'],
                'retained_id':match['id'], 'retained_title':match['title'], 'reason':reason,
                'lexical_similarity':round(similarity(row['text'],match['text']),6),
                'interpretation':'Source/topic concentration control is not a finding of identical text.'})
        else:
            kept.append(row)
            source_seen[source] = row
            if cluster:
                cluster_seen[cluster] = row
    kept.sort(key=lambda r:order[r['id']])
    removed.sort(key=lambda r:order[r['id']])
    return kept, removed


def build():
    cycle = Path('data/petlm/experiments/generation-repair-v1')
    source = cycle/'licensed-wikipedia-inspected-r6'
    output = cycle/'licensed-wikipedia-diverse-r7'
    if output.exists():
        raise FileExistsError(output)
    expected = '4a32d9e919776c85b1decf19ee6c5f9534cb2e2ad13e3662aa4980ff2e67aa38'
    if checksum(source/'manifest.json') != expected:
        raise ValueError('Selection policy belongs to another source snapshot')
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        if checksum(source/name) != digest:
            raise ValueError(f'Source snapshot changed: {name}')
    rows = json.loads((source/'review.json').read_text(encoding='utf-8'))['rows']
    if len(rows)!=23:
        raise ValueError('Unexpected source review length')
    # Original review numbers are bound by the checked manifest above.
    ids = {i:r['id'] for i,r in enumerate(rows,1)}
    clusters = {ids[i]:'whale_body_description' for i in (13,15)}
    clusters.update({ids[i]:'park_layout_and_history' for i in (2,11,12)})
    preferred = [ids[i] for i in (1,7,12,15,17)]
    kept, removed = reduce_rows(rows, clusters, preferred)
    output.mkdir()
    shutil.copy2(Path(__file__),output/'selection-source.py')
    counts,tokens = Counter(),Counter()
    for split in ('train','validation','test'):
        path=output/(split+'-candidates.jsonl')
        if split=='test':
            shutil.copy2(source/path.name,path)
            counts[split]=manifest['candidate_counts'][split]
            tokens[split]=manifest['candidate_tokens_with_eos'][split]
            continue
        with path.open('x',encoding='utf-8',newline='\n') as handle:
            for row in kept:
                if row['split']==split:
                    counts[split]+=1
                    tokens[split]+=row['tokens_with_eos']
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(output/'review.json',{'status':'pending_new_human_review',
        'scope':'All reduced train/validation rows only; reserved test not displayed.', 'rows':kept})
    original_numbers={r['id']:i for i,r in enumerate(rows,1)}
    lines=['# 精简后的中文段落审核清单','',
        f'共 {len(kept)} 条：训练候选 {counts["train"]} 条，验证候选 {counts["validation"]} 条。',
        '同一来源条目最多保留一段；相近鲸类外形描述、公园景观与沿革介绍各保留一段。',
        '原正文及分区保持不变。来源或主题相近不等于逐字重复；删选原因见 removal-report.md。',
        '共同来源与署名：中文维基百科贡献者；正文许可 CC BY-SA 4.0。',
        '许可说明：https://creativecommons.org/licenses/by-sa/4.0/；每条附原修订与贡献者历史。',
        '人工审核仍未完成。逐条检查自然度、完整性、上下文、事实、内容适宜性、隐私与第三方引用例外。',
        '旧编号便于与之前的 23 条清单核对；旧清单已归档。测试内容未展示，训练仍关闭。','']
    for i,row in enumerate(kept,1):
        lines += [f'## R{i:03d} · {row["split"]} · {row["title"]}（原 N{original_numbers[row["id"]]:03d}）',
            f'ID：{row["id"]}',f'原修订：{row["origin"]}',f'贡献者历史：{row["history_url"]}',
            '',row['text'],'','人工结论：待审核','审核人：','理由：','']
    (output/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    exact_pairs = sum(normalized(a['text'])==normalized(b['text']) for i,a in enumerate(rows) for b in rows[:i])
    audit={'kind':'review_redundancy_audit_v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
        'source_manifest_sha256':expected,'original_review_rows':len(rows),
        'normalized_exact_duplicate_pairs':exact_pairs,
        'near_duplicate_threshold':.75,
        'normalized_fourgram_overlap_pairs':sum(similarity(a['text'],b['text'])>=.75 for i,a in enumerate(rows) for b in rows[:i]),
        'repeated_source_titles':{k:v for k,v in Counter(r['title'] for r in rows).items() if v>1},
        'kept_rows':len(kept),'removed_rows':len(removed),'removed':removed,
        'manual_description_clusters':clusters,'preferred_representatives':preferred,
        'manual_preference_rationale':'Keep general explanations over dense name lists; one whale anatomy example and one park-layout example. No blind output or score used.',
        'text_rewritten':False,'partitions_changed':False,'reserved_test_sha256_unchanged':checksum(source/'test-candidates.jsonl')==checksum(output/'test-candidates.jsonl'),
        'human_approval':False,'training_started':False,
        'limits':'Lexical checks and listed topic clusters do not certify complete semantic novelty.'}
    save(output/'redundancy-audit.json',audit)
    report=['# 精简记录','',f'原清单 {len(rows)} 条 → 当前 {len(kept)} 条；正文逐字重复对数 {exact_pairs}。',
        '本次主要减少来源与主题集中，不把不同年代、不同对象的段落误称为完全重复。','']
    reason_labels={'same_source_article_cap':'同一条目只保留一段','similar_topic_and_description_cap':'相近主题及描述方式限留一段','lexical_near_duplicate':'正文相似度超过阈值'}
    for row in removed:
        report.append(f'- 原 N{original_numbers[row["id"]]:03d}《{row["title"]}》：{reason_labels[row["reason"]]}；保留原 N{original_numbers[row["retained_id"]]:03d}《{row["retained_title"]}》。')
    (output/'removal-report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    result=dict(manifest,kind='source_and_topic_reduced_candidate_subset_v1',
        parent_candidate_path=str(source),parent_candidate_manifest_sha256=expected,
        status='pending_human_review_insufficient_scale_and_validation',
        candidate_counts=dict(counts),candidate_tokens_with_eos=dict(tokens),review_rows=len(kept),
        assistant_reviewed_train_validation_rows=len(rows),assistant_held_rows=len(removed),
        source_cap_per_review=1,topic_policy='Listed similar-description clusters capped at one; broad topic equality alone is not treated as duplication.',
        training_ready=False,training_started=False,human_review_completed=False)
    result['files']={p.name:checksum(p) for p in output.iterdir() if p.is_file()}
    save(output/'manifest.json',result)
    print(json.dumps({'counts':dict(counts),'tokens':dict(tokens),'removed':len(removed),'exact_duplicate_pairs':exact_pairs},ensure_ascii=False))


if __name__=='__main__':
    build()
