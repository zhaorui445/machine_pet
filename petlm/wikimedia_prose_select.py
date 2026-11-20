"""Narrow verified Wikimedia candidates by genre, context and observed defects."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse, unquote, parse_qs

from .prose_supplement import save
from .run import checksum
from .wikimedia_prose import group_key, fingerprints, normalize

BOOKS = {'自然科学','高中数学','初中物理','初中化学','普通生物学','生态学和生物圈',
         '温度与温标','食谱','猫','半导体器件和技术','统计学','读书方法','学习方法'}
TOPICS = {
    'food': ('饮食','美食','点心','米饭','面食','餐馆','早餐','小吃','蔬菜','水果','茶叶','咖啡'),
    'daily_travel': ('公交','步行','散步','公共交通','地铁','自行车','图书馆'),
    'nature': ('公园','花园','湖泊','植物','森林','河流','气候','四季','春季','夏季','秋季','冬季','温度','降雨'),
}
CONTEXT = re.compile(r'^(?:而|他|她|它|这幅|这些|这种|该|此|由于)|先前|上述|前述|图中|下文|上文|本节|本章|如下|如前|如后|\(注|注[一二三四五六七八九十]+|前面|后面|接下来')
DEFECT = re.compile(r'的的|了了|确定确保|主要为.{0,15}与该|为.{0,25}最奇妙景象|主要的业务为|属于综合型|漫波|笃定|每个个体|更来得|还有大量的可看|供应量很少|夏季春分|贵格七世利于|\u200b|(?<=[\u3400-\u9fff]) (?=[\u3400-\u9fff])')
QUOTE_OR_PROMO = re.compile(r'摘自|选自|转载|转自|版权所有|版權所有|品牌|知名企业|国家发展|政府|集团|生产基地|工业园|商贸|古代|皇帝|王朝|教宗|神学|圣灵|基督|疗法|寄生虫|潜水|攀岩|登山|高压|电源|电气|火山|温泉')


def classify(row):
    title, text = row['title'], row['text']
    if CONTEXT.search(text):
        return None, 'context_dependency'
    if DEFECT.search(text) or re.search(r'\([一二三四五六七八九十]+\)|\b[1-9]\.\s', text):
        return None, 'observed_defect_or_flattened_list'
    if QUOTE_OR_PROMO.search(title+'\n'+text):
        return None, 'outside_current_natural_prose_scope'
    if row['source'] == 'zhwikibooks':
        if title.split('/')[0] in BOOKS:
            return 'learning_and_science', None
        # Psychology is a large book; only educational, nonclinical chapters.
        if title.split('/')[0] == '超普通心理学' and re.search(r'/(学习|记忆|语言)(?:/|$)', title):
            if re.search(r'嗅觉|香氛|催眠|实验|受试|研究发现|研究显示|大脑|神经|海马|情欲|恐惧|虐待', text):
                return None, 'specialized_claim_outside_pilot'
            return 'learning_and_memory', None
        return None, 'book_not_in_current_genre_selection'
    hits = {k:sum(word in text for word in words) for k,words in TOPICS.items()}
    genre = max(hits, key=hits.get)
    return (genre, None) if hits[genre] >= 2 else (None, 'not_daily_or_nature_prose')


def origin_group(url):
    parsed = urlparse(url)
    hosts = {'zh.wikibooks.org':'zhwikibooks','zh.m.wikibooks.org':'zhwikibooks',
             'zh.wikivoyage.org':'zhwikivoyage','zh.m.wikivoyage.org':'zhwikivoyage'}
    source = hosts.get(parsed.hostname)
    if not source:
        return None
    title = unquote(parsed.path.split('/wiki/',1)[1]) if '/wiki/' in parsed.path else parse_qs(parsed.query).get('title',[''])[0]
    return group_key(source, title.replace('_',' ')) if title else None


def select(source, base, output):
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if meta['kind'] != 'wikimedia_natural_chinese_candidate_v1':
        raise ValueError('Wrong candidate source')
    parent = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if checksum(base/'manifest.json') != meta['parent_manifest_sha256']:
        raise ValueError('Parent snapshot changed')
    provenance = base/'provenance.jsonl'
    if checksum(provenance) != parent['files']['provenance.jsonl']:
        raise ValueError('Parent provenance changed')
    parent_groups = set()
    with provenance.open(encoding='utf-8') as stream:
        for line in stream:
            key = origin_group(json.loads(line).get('origin',''))
            if key:
                parent_groups.add(key)
    kept, rejected = [], Counter()
    for split in ('train','validation','test'):
        path = source/f'{split}-candidates.jsonl'
        if checksum(path) != meta['files'][path.name]:
            raise ValueError('Input candidate changed')
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['split'] != split:
                raise ValueError('Split mismatch')
            genre, why = classify(row)
            if row['split_group'] in parent_groups:
                why = 'parent_source_book_or_city_group'
            if why:
                rejected[why] += 1
            else:
                kept.append(dict(row, suggested_genre=genre))
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output/'selection-source.py')
    shutil.copy2(source/'ATTRIBUTION.md', output/'ATTRIBUTION.md')
    counts, tokens, genres, partitions = Counter(), Counter(), Counter(), {}
    for split in ('train','validation','test'):
        partitions[split] = {'hashes':set(),'groups':set()}
        with (output/f'{split}-candidates.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for row in kept:
                if row['split'] != split:
                    continue
                counts[split] += 1
                tokens[split] += row['tokens_with_eos']
                genres[f'{split}/{row["suggested_genre"]}'] += 1
                partitions[split]['hashes'].update(fingerprints(row['text']))
                partitions[split]['groups'].add(row['split_group'])
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    checks = {}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for key in ('hashes','groups'):
            checks[f'{a}_{b}_{key}_overlap'] = len(partitions[a][key]&partitions[b][key])
    if any(checks.values()):
        raise ValueError('Cross-split overlap')
    sample = []
    for split in ('train','validation'):
        for genre in sorted({r['suggested_genre'] for r in kept}):
            pool = sorted((r for r in kept if r['split']==split and r['suggested_genre']==genre),key=lambda r:r['id'])
            used, selected = set(), []
            for row in pool:
                if row['split_group'] not in used:
                    selected.append(row)
                    used.add(row['split_group'])
                if len(selected)>= (8 if split=='train' else 3):
                    break
            if len(selected)<(8 if split=='train' else 3):
                selected += [r for r in pool if r['id'] not in {s['id'] for s in selected}][:(8 if split=='train' else 3)-len(selected)]
            sample.extend(selected)
    save(output/'review.json',{'scope':'New stratified samples only; not all rows', 'status':'pending_human_review',
        'rows':[dict(r,human_decision=None,reviewer=None,review_notes='') for r in sample]})
    lines = ['# 新来源精选段落：待人工抽样复核','',
        'CC BY-SA 4.0；原文清洗、繁简转换，未补写或拼接。旧批次审核不适用。',
        '检查自然度、语法、能否独立理解、事实与建议、隐私和版权例外。填写通过/拒绝/待定及理由。',
        '本表仅抽样，不能表述为全量逐条审核；验证文本不能训练；测试正文不展示。','']
    for i,row in enumerate(sample,1):
        lines += [f'## S{i:03d} · {row["split"]} · {row["suggested_genre"]}',
                  f'标题：{row["title"]} / {row["section"]}', f'ID：{row["id"]}',
                  f'原文：{row["origin"]}',f'贡献者：{row["history_url"]}','',row['text'],'',
                  '人工结论：待审核','审核人：','理由：','']
    (output/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    save(output/'verification.json',dict(checks,parent_source_groups_checked=len(parent_groups),
        source_integrity_verified=True, parent_pretraining_overlap_check_inherited=True,
        no_frozen_benchmark_used=True, all_rows_pending_human_review=True))
    report = {'kind':'focused_licensed_natural_chinese_v1','status':'pending_new_human_sample_review',
        'input':str(source),'input_manifest_sha256':checksum(source/'manifest.json'),
        'license':meta['license'],'counts':dict(counts),'tokens_with_eos':dict(tokens),
        'genres':dict(genres),'rejected':dict(rejected),'review_samples':len(sample),
        'training_ready':False,'training_started':False,'human_review_completed':False,
        'physical_actuation':'disabled','split_policy':meta['split_policy'],
        'limitations':meta['limitations']+['Genre filtering and assistant inspection are not human approval.'],
        'files':{p.name:checksum(p) for p in output.iterdir() if p.is_file()}}
    save(output/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ('counts','tokens_with_eos','genres','review_samples')},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    select(args.source,args.base,args.output)


if __name__=='__main__':
    main()
