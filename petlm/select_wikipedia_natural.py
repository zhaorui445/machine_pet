"""Conservative topical selection; keeps source words and original page splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from .run import checksum
from .prose_supplement import save
from .wikimedia_prose import fingerprints


TOPICS = {
    'nature':r'^(?:牛|羊|马|猫|犬|鱼|鸟|树|花|草|竹)$|植物|动物|昆虫|蝴蝶|蜜蜂|蚂蚁|蚯蚓|蜗牛|蜻蜓|松鼠|海豚|鲸|企鹅|熊猫|短尾猫|叶绿|光合|生态|生物多样性|湿地|珊瑚|森林|红树林|落叶|常绿',
    'science':r'天文|宇宙|恒星|银河|星系|行星|月球|太阳系|日食|月食|地球|气候|季风|海洋|潮汐|地质|岩石|矿物|水循环|彩虹|光学|声学|光线|声音|浮力|重力|磁场|电磁|摩擦|蒸发|凝结',
    'learning':r'学习|阅读|写作|图书馆|书籍|识字|教育|教学|实验|观察|博物馆|记忆|思考|逻辑|语言|语法|词汇|标点|拼音|手语',
    'arts':r'乐谱|乐器|音乐|钢琴|提琴|吉他|竖琴|口琴|笛|古筝|琵琶|二胡|绘画|美术|书法|陶瓷|编织|雕塑|摄影|舞蹈|花灯|灯笼|纸艺|剪纸|折纸',
    'everyday':r'厨房|烹饪|烹调|面包|馒头|米饭|面条|豆腐|蔬菜|水果|园艺|种植|散步|步行|自行车|游泳|足球|篮球|羽毛球|乒乓球|公园|花园|庭院|登山|露营|回收|节水|垃圾分类|雨伞|风筝',
}
EXCLUDE = re.compile(r'战争|战役|军队|军人|士兵|国防|北伐|侵略|宗教|基督|上帝|圣经|佛教|教义|判决|诉讼|刑事|证券|贷款|投资|心理医生|疾病|治疗|毒物|中毒|饮毒|强奸|食人|政党|总统|总理|特首|选举|中国共产党|国民党|动画|漫画|角色|剧情|查克拉|写轮眼|监督|教会|清真|伊斯兰')
CONTEXT = re.compile(r'^(?:在此|对于|就此|又|再|依照同样|按照上述|至于|其中|而|故|[（(])|如上|上述|上文|下文|前面|后面提到|这个两个|前状前肢|利益思想|论坛次|沒有中共召开|没有中共召开')


def select_reason(row):
    title = row['title']
    topic = next((name for name, pattern in TOPICS.items() if re.search(pattern, title)), None)
    if topic is None:
        return None, 'not_priority_explanatory_topic'
    if EXCLUDE.search(title+' '+row['section']+' '+row['text']):
        return None, 'outside_current_explanatory_scope'
    if CONTEXT.search(row['text']):
        return None, 'context_dependency_or_observed_defect'
    if len(re.findall('所以', row['text'])) > 1:
        return None, 'repeated_connective_requires_review'
    return topic, None


def build(source, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if checksum(source/name) != expected:
            raise ValueError('Parent candidate snapshot changed')
    # Prior assistant negatives are development observations, not human scores.
    review_root = source.parent/'licensed-wikipedia-paragraphs-r3'
    reviewed = json.loads((review_root/'assistant-sample-review.json').read_text(encoding='utf-8'))
    negatives = {r['id'] for r in reviewed['rows'] if r['assistant_decision'] != 'candidate'}
    rows, rejected, themes = [], Counter(), Counter()
    for split in ('train','validation','test'):
        with (source/(split+'-candidates.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                topic, why = select_reason(row)
                if row['id'] in negatives:
                    why = 'previous_assistant_quality_hold'
                if why:
                    rejected[why] += 1
                else:
                    rows.append(dict(row, topical_stratum=topic))
                    themes[split+'/'+topic] += 1
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output/'selection-source.py')
    counts, tokens = Counter(), Counter()
    for split in ('train','validation','test'):
        with (output/(split+'-candidates.jsonl')).open('x', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                if row['split'] == split:
                    counts[split] += 1
                    tokens[split] += row['tokens_with_eos']
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    review = sorted((r for r in rows if r['split'] != 'test'), key=lambda r:(r['split'],r['id']))
    save(output/'review.json', {'status':'pending_new_human_review',
        'scope':'All selected train and validation paragraphs; excludes reserved test.',
        'rows':[dict(r, decision=None, reviewer=None, notes='') for r in review]})
    batch_dir = output/'review-batches'
    batch_dir.mkdir()
    index = ['# 自然中文候选：人工审核入口', '',
        '这些是从中文维基百科原始段落中按科普、学习、艺术、生活主题筛出的新候选，尚未人工批准。',
        '标题关键词只是主题初筛，不能保证内容自然、事实准确或不存在第三方权利例外。',
        '每条检查自然度、完整性、上下文相关性、事实及内容适宜性、个人信息和引用来源。',
        '请填通过/拒绝/待定及理由；旧批次审批不覆盖本批。保留测试正文不在审核表中。',
        f'本表共 {len(review)} 条，全部 train/validation 候选均在下面的分批文件中。', '']
    for start in range(0, len(review), 100):
        lines = [f'# 审核批次 {start//100+1:03d}', '', '请逐条填写，不按整批默认通过。', '']
        for i,row in enumerate(review[start:start+100],start+1):
            lines += [f'## N{i:04d} · {row["split"]} · {row["title"]}',
                f'ID：{row["id"]}', f'来源：{row["origin"]}', f'作者历史：{row["history_url"]}',
                f'许可：{row["license"]} · {row["license_url"]}',
                f'分节：{row["section"]}', '', row['text'], '',
                '人工结论：待审核', '审核人：', '理由：', '']
        name=f'batch-{start//100+1:03d}.md'
        (batch_dir/name).write_text('\n'.join(lines), encoding='utf-8')
        index.append(f'- [批次 {start//100+1:03d}：N{start+1:04d}–N{min(start+100,len(review)):04d}](review-batches/{name})')
    (output/'review.md').write_text('\n'.join(index), encoding='utf-8')
    result = dict(manifest, kind='topically_selected_wikipedia_paragraph_candidates_v1',
        status='pending_new_human_review_not_training_ready',
        parent_candidate_path=str(source), parent_candidate_manifest_sha256=checksum(source/'manifest.json'),
        candidate_counts=dict(counts), candidate_tokens_with_eos=dict(tokens),
        topical_counts=dict(themes), selection_rejections=dict(rejected),
        review_rows=len(review), assistant_sample_rows=0,
        prior_assistant_negative_record_sha256=checksum(review_root/'assistant-sample-review.json'),
        text_rewritten=False, source_splits_preserved=True,
        training_ready=False, human_review_completed=False, training_started=False)
    result['limitations'] += ['Title keyword strata are only routing heuristics; not human quality certification.']
    result['files']={p.relative_to(output).as_posix():checksum(p) for p in output.rglob('*') if p.is_file()}
    save(output/'manifest.json', result)
    print(json.dumps({k:result[k] for k in ('candidate_counts','candidate_tokens_with_eos','review_rows','topical_counts','selection_rejections')},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.output)
