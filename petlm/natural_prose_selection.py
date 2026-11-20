"""Prioritize whole natural-prose candidates; never approve or train them."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

from .prose_repair import hashes
from .prose_supplement import save, DOMAINS
from .run import checksum


OFFICIAL = re.compile(r'召开|出席|列席|贯彻|部署|党委|党支部|党组|常委|政协|纪委|机关事务|工作会议|座谈会|主题教育|签约仪式|推介会|营商环境|表态发言|领导讲话')
MARKETING = re.compile(r'投资兴业|商机|旗下品牌|一站式|行销|营销|赠送|赠品|限量|免费领取|VX|微.?信号|回复关键词|强势入驻|诚邀|热线|招商', re.I)
TEMPLATE = re.compile(r'范文|毕业总结|工作总结|心得体会范文|201[xX]|20[xX]{2}|第[一二三四五六七八九十]+篇|本文来自|原标题|责编[:：]|责任编辑[:：]|本报讯|评论精选')
FRAGMENT = re.compile(r'的的|了了|充分挥发|工作点特|党建工创新|开始使将|(?:同比|增长|下降)[^。！？\n]{0,4}%|(?:投资|收入|支出)\s*(?:亿|万)元')
DAILY = ('做饭', '洗菜', '切菜', '煮饭', '厨房', '食材', '早餐', '晚饭', '晚餐', '散步', '家务', '打扫房间', '洗衣', '晾衣', '买菜', '浇水', '种花', '收拾', '书包', '放学', '公交', '回家', '出门', '遛狗')
LEARNING = ('做题', '复习', '预习', '笔记', '错题', '背诵', '练习', '朗读', '阅读', '单词', '作业', '课文', '记忆', '理解', '学习方法')
SCIENCE = ('水蒸气', '蒸发', '凝结', '温度', '湿度', '空气', '光照', '阳光', '植物', '叶片', '种子', '根系', '光合作用', '摩擦', '重力', '浮力', '磁场', '声波', '地球', '月球', '实验', '观察')
PROCESS = ('因为', '所以', '因此', '如果', '首先', '然后', '接着', '最后', '原来', '发现', '为了', '当时', '后来', '开始', '逐渐', '这样', '例如', '比如', '由于')
RESIDUAL = re.compile(r'万博|新宝gg|金沙澳门|提款|实名认证|找富婆|穿越|前世|灵鼎|五行鼎|枪战|不想活|轻生|遗书|风水|运势|命相|密斯|犯警|原原料|[?？!！]。|经融危机|变成得|厂家|供暖设备|市场消费', re.I)


def classify(text):
    for name, pattern in [('institutional_report', OFFICIAL), ('promotion', MARKETING),
                          ('template_or_news_aggregation', TEMPLATE), ('observed_text_defect', FRAGMENT),
                          ('residual_spam_corruption_or_outside_pilot_scope', RESIDUAL)]:
        if pattern.search(text):
            return None, name, {}
    hits = {name: [word for word in words if word in text]
            for name, words in [('daily', DAILY), ('learning', LEARNING), ('explanation', SCIENCE)]}
    process = [word for word in PROCESS if word in text]
    genre = max(hits, key=lambda name: len(hits[name]))
    # Keyword matching prioritizes review only, not semantic acceptance.
    if len(hits[genre]) < 2 or len(process) < 2:
        return None, 'insufficient_topic_and_process_evidence', {'topic_hits': hits, 'process_hits': process}
    return genre, None, {'topic_hits': hits, 'process_hits': process}


def build(sources, output):
    if output.exists():
        raise FileExistsError(output)
    inputs, rows, rejected = [], [], Counter()
    for source in sources:
        manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if manifest['kind'] != 'new_fineweb_prose_supplement_audit_v1':
            raise ValueError('Expected verified whole-prose supplement')
        inputs.append({'path': str(source), 'manifest_sha256': checksum(source/'manifest.json')})
        for split in ('test', 'validation', 'train'):
            for group in ('source-restricted', 'other-sources-hold'):
                path = source/f'{group}-{split}.jsonl'
                if checksum(path) != manifest['files'][path.name]:
                    raise ValueError(f'Input changed: {path}')
                for line in path.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    if row['split'] != split:
                        raise ValueError('Split mismatch')
                    genre, why, signals = classify(row['text'])
                    if why:
                        rejected[why] += 1
                    else:
                        host = (urlparse(row['origin']).hostname or '').lower()
                        source_tier = ('recorded_domain_priority' if any(host == d or host.endswith('.'+d) for d in DOMAINS)
                            and not any(part in ('bbs', 'forum', 'blog') for part in host.split('.')) else 'source_review_hold')
                        rows.append(dict(row, suggested_genre=genre, selection_signals=signals,
                            source_tier=source_tier,
                            parent_candidate_file=str(path), human_decision=None, reviewer=None,
                            source_rights_status='original_page_not_verified', training_eligible=False))
    # Preserve all existing split assignments; held-out wins cross-source collisions.
    rows.sort(key=lambda r: ({'test': 0, 'validation': 1, 'train': 2}[r['split']], r['id']))
    kept, seen = [], set()
    for row in rows:
        fingerprints = hashes(row['text'])
        if fingerprints & seen:
            rejected['cross_input_exact_prefix_or_paragraph_duplicate'] += 1
        else:
            seen.update(fingerprints)
            kept.append(row)
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output/'selection-source.py')
    counts, tokens, genres, source_tiers = Counter(), Counter(), Counter(), Counter()
    for split in ('train', 'validation', 'test'):
        with (output/f'{split}-candidates.jsonl').open('x', encoding='utf-8', newline='\n') as f:
            for row in kept:
                if row['split'] == split:
                    counts[split] += 1
                    tokens[split] += row['tokens_with_eos']
                    genres[f'{split}/{row["suggested_genre"]}'] += 1
                    source_tiers[f'{split}/{row["source_tier"]}'] += 1
                    f.write(json.dumps(row, ensure_ascii=False)+'\n')
    review = sorted([r for r in kept if r['split'] != 'test'], key=lambda r: (r['source_tier'], r['split'], r['id']))
    save(output/'review.json', {'status': 'pending_human_and_source_review', 'rows': review})
    lines = ['# 自然中文定向候选：待审核', '',
        '完整保留已清洗文档，没有改写病句、摘拼段落或移动训练/验证划分。',
        '关键词仅用于排序筛选；不证明自然度、事实准确、隐私或原网页许可。',
        '人工核验：来源与许可、自然完整、语法、上下文、隐私和内容适宜性。逐条填写通过/拒绝/待定及原因。',
        '旧批次审核不适用。test 文本不展示，validation 不能进入训练。', '']
    for i, row in enumerate(review, 1):
        lines += [f'## P{i:03d} · {row["split"]} · {row["suggested_genre"]}',
                  f'ID：{row["id"]}', f'来源：{row["origin"]}', f'来源分级：{row["source_tier"]}（均未确认原网页许可）', '', row['text'], '',
                  '人工结论：待审核', '原网页许可证据：待核验', '审核人：', '理由：', '']
    (output/'review.md').write_text('\n'.join(lines), encoding='utf-8')
    report = {'kind': 'targeted_whole_natural_prose_selection_v1', 'inputs': inputs,
        'status': 'not_training_ready', 'counts': dict(counts), 'tokens_with_eos': dict(tokens),
        'suggested_genres': dict(genres), 'rejected': dict(rejected), 'review_rows': len(review),
        'source_tiers': dict(source_tiers),
        'text_modified': False, 'source_rights_verified': False, 'human_review_completed': False,
        'training_started': False, 'physical_actuation': 'disabled',
        'limitations': ['Genre signals are heuristic; all retained rows still need text and provenance review.',
            'Parent full/prefix/paragraph exclusion inherited from input snapshots; no historical semantic independence claim.',
            'No direct training export or tokenizer binary creation before quality and review gates.'],
        'files': {p.name: checksum(p) for p in output.iterdir() if p.is_file()}}
    save(output/'manifest.json', report)
    print(json.dumps({k: report[k] for k in ('counts', 'tokens_with_eos', 'suggested_genres', 'review_rows')}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == '__main__':
    main()
