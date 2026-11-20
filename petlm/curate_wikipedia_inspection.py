"""Materialize recorded assistant inspection; never grant human approval."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from .run import checksum
from .prose_supplement import save


# Decisions refer to the exact hash-bound r5 review order inspected in full.
KEEP = {1,2,3,7,9,13,14,17,20,37,39,40,41,53,61,63,67,74,79,86,90,110,118}
REASONS = {
4:'位于重复，末句过长且结构绕乱。',5:'多处译文式搭配不自然，例如发生一些灾害周边林区。',
6:'分类成本与最终品质的逻辑和语句连接欠清楚。',8:'旦另一些为错字，末句结构错误。',
10:'科学猜测和事实交织，需专门事实核对。',11:'赛季之初未给出年份，段落上下文不足。',
12:'球队名单密集且带较强评价，不优先用于本轮基础说明文。',15:'依赖右方第一张图及另外两幅缺失图。',
16:'以这三张图开头，缺失图示依赖。',18:'专业理论过密，并有不自然译法；需专门科学审校。',
19:'缺少开头赛季背景，含比赛判罚争议和冗长过程。',21:'这一新的语言依赖前文，部分译词生硬。',
22:'第一句缺主语；绝对化火灾处理表述不适合直接训练。',23:'制度争议与历史政策说明，不优先作为学习方法说明。',
24:'具体行政争议叙事，不是日常或科普说明。',25:'尽管开头依赖既有投入背景。',26:'含语句连接缺失及法条/教育争议背景。',
27:'亚冠小组出现疑似出线错字，球队名也需核验。',28:'在用等搭配不顺，年代和考古细节需核验。',
29:'公元922年与宋太宗年号存在明显时间矛盾，需核对原文。',30:'但对于...情况下结构不顺，制度断言需核验。',
31:'不段错字及多处片段句。',32:'差强人意用法和叙事连贯性待核验；赛事细节过密。',
33:'曾在此承接上文，含拔选等疑似错字。',34:'受集错字且涉及特工叙事，不作为本轮优先内容。',
35:'外籍球员球员重复、日期段落缺谓语。',36:'依赖右图和图中内容。',38:'回顾整个赛季缺年份，且含赞誉性判断。',
42:'一个有...身体之后缺中心词，译文句法有缺陷。',43:'这张表格及技术段落依赖未提供表格。',
44:'清未疑似清末错字，列表式正文。',45:'专业术语密集且句法破碎，需科学审校。',
46:'拥做等词语与句法不够规范。',47:'每只车队量词不当，内容为角色长列表。',48:'未说明赛季，保守战术与评价需要上下文。',
49:'不但...很少有不拿来临摹学习句式未完整收束。',50:'强宣传性表述和极值断言需来源核查。',
51:'珊瑚误命中自然主题，实际为海战内容。',52:'音乐误命中，实际为剧情并依赖前文。',
54:'积分分重复，赛季叙事需要上下文。',55:'基本粒子和力与恒星等概念混杂，译文不连贯。',
56:'滚轴溜冰喜爱缺少人的指称。',57:'初期与工行背景未交代，现有藏品数需时间限定。',
58:'同年开头且引文截断，属于政治争议叙事。',59:'林木种类、因果和年代措辞需要生态专业核查。',
60:'包裹在神秘之中、汲取强大的能量等译法不自然。',62:'球队命名政策采访，引语与主体内容不优先。',
64:'重复宣布和悬挂悬挂。',65:'缺少年份的连续赛事记录。',66:'法定管理权限说明需独立法律审核。',
68:'特别是承接上文，数据断言需核查。',69:'使了语病，且赛季背景缺失。',70:'首场比赛和转会时间背景未交代。',
71:'前后信息反复回环，技术解释难以独立理解。',72:'现行租金和地方服务细节，不适合作为稳定说明材料。',
73:'投灾疑似错字，尚未恢复等断言无日期。',75:'现代物种估计与变化趋势的强断言需要科学审校。',
76:'因为重复、则是认为是等明显语病。',77:'代表的等句尾不完整，近年无日期。',
78:'大量专业演化断言及唯一等绝对措辞需要核查。',80:'地方安保及派车安排时效性强，不优先。',
81:'科学阶段叙事与译文表达不清。',82:'体育历史总结，不优先用于当前说明文；赛制断言待核验。',
83:'目标材料比例定义方向不清，末句绝对化。',84:'珊瑚误命中，实际为海战战果。',
85:'图书馆选址争议，陈德胜又表示承接上文。',87:'结尾崩落成缺少宾语。',
88:'大量重复保留，且为具体工程进度。',89:'回顾整个赛季缺年份，叙事依赖前文。',
91:'和我们今天看到我们周围等重复且逻辑破碎。',92:'跨度过大的科学因果叙事、富养化用词需核验。',
93:'体育赛季总结，不优先作为当前说明文。',94:'核反应解释和译文句法需专门科学审校。',
95:'教育误命中，实际为组织解散及政治争议。',96:'氦原子核被描述为质子等表述需核查，部分用语生硬。',
97:'巨大成本使回收业者没有利润可能等语病。',98:'两位拒绝保育界学者语句缺损，并涉及公约及政治争议。',
99:'另一方面依赖上文，且技术叙述密集。',100:'技术译文生硬，发光却更低等措辞需校对。',
101:'包含强历史断言与较多引文，先保留待事实核验。',102:'王即王铎缺前文，句法片段与师承关系待核验。',
103:'商品宣传内容。',104:'经济因果断言不明，且部分译文不自然。',105:'体育历史过程较冗长，不优先当前说明文。',
106:'引用研究及模型解释需要技术和上下文核验。',107:'少样等用语欠规范，生态断言和政治评价混合。',
108:'范围宽泛的人群语言断言缺时间限定，需核验。',109:'恒星的核心就成为只有中子的致密球体重复。',
111:'太初核合成时程与重元素表述需科学核验。',112:'未来工程安排和自然语言不够精炼，不作为独立语言验证材料。',
113:'拥有的过程及虽然...但结构生硬。',114:'曾经做过...末尾缺少动作对象，表述需要校对。',
115:'术语仙王变星和历史叙述疑似错误。',116:'类型罗列和主题型的主题公园表达重复。',
117:'辐辙错词，且科学史表述需要核对。',
}


def build():
    cycle=Path('data/petlm/experiments/generation-repair-v1')
    source=cycle/'licensed-wikipedia-natural-r5'
    output=cycle/'licensed-wikipedia-inspected-r6'
    if output.exists():
        raise FileExistsError(output)
    expected='de80a8b4e59494dcbc32975b480d0899fbd7b42a780c4e2326372ea566953383'
    if checksum(source/'manifest.json') != expected:
        raise ValueError('Inspection mapping belongs to a different snapshot')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        if checksum(source/name) != digest:
            raise ValueError('Inspected snapshot changed')
    rows=json.loads((source/'review.json').read_text(encoding='utf-8'))['rows']
    if len(rows)!=118 or set(KEEP)|set(REASONS)!=set(range(1,119)) or set(KEEP)&set(REASONS):
        raise ValueError('Incomplete assistant inspection mapping')
    output.mkdir()
    inspected=[]
    kept=[]
    for i,row in enumerate(rows,1):
        decision='candidate_for_human_review' if i in KEEP else 'hold'
        inspected.append({'review_index':i,'id':row['id'], 'split':row['split'],
            'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
            'assistant_decision':decision,
            'notes':REASONS.get(i,'段落相对完整、连贯，建议进入新的人工复核；具体事实与引用许可未独立核实。')})
        if i in KEEP:
            kept.append(dict(row, assistant_quality_status=decision))
    save(output/'assistant-review.json', {'kind':'assistant_quality_review_not_human_approval',
        'recorded_utc':datetime.now(timezone.utc).isoformat(), 'reviewed_rows':118,
        'source_manifest_sha256':expected,'test_text_read':False,
        'human_approval':False,'rows':inspected})
    shutil.copy2(Path(__file__),output/'selection-source.py')
    counts,tokens=Counter(),Counter()
    for split in ('train','validation','test'):
        subset=[r for r in kept if r['split']==split]
        if split=='test':
            # Copy opaque reserved material; do not inspect/tune on its contents.
            shutil.copy2(source/'test-candidates.jsonl',output/'test-candidates.jsonl')
            counts[split]=manifest['candidate_counts'][split]
            tokens[split]=manifest['candidate_tokens_with_eos'][split]
            continue
        with (output/(split+'-candidates.jsonl')).open('x',encoding='utf-8',newline='\n') as handle:
            for row in subset:
                counts[split]+=1
                tokens[split]+=row['tokens_with_eos']
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(output/'review.json',{'status':'pending_new_human_review',
        'scope':'All 23 assistant-selected train/validation paragraphs only; test excluded.',
        'rows':kept})
    lines=['# 新增中文段落：人工复核清单','',
        '助手已检查上一级候选中的全部 118 条训练/验证段落；本表为建议继续人工复核的 23 条。',
        '人工审核状态仍为待审核。旧批准不覆盖本批；保留测试内容未展示。',
        '来源：中文维基百科，CC BY-SA 4.0；每条保留修订与贡献者链接，正文未改写。',
        '检查自然完整、上下文独立、事实、内容适宜性、隐私和第三方引用许可；填通过/拒绝/待定及理由。',
        '这些少量段落尚不能支持开训，验证仅一段，独立性与代表性规模不足。','']
    for i,row in enumerate(kept,1):
        lines += [f'## N{i:03d} · {row["split"]} · {row["title"]}',f'ID：{row["id"]}',
            f'来源：{row["origin"]}',f'贡献者历史：{row["history_url"]}',
            f'许可：{row["license"]} · {row["license_url"]}','',row['text'],'',
            '人工结论：待审核','审核人：','理由：','']
    (output/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    result=dict(manifest, kind='assistant_inspected_wikipedia_paragraph_subset_v1',
        status='pending_human_review_insufficient_scale_and_validation',
        parent_candidate_path=str(source), parent_candidate_manifest_sha256=expected,
        candidate_counts=dict(counts),candidate_tokens_with_eos=dict(tokens),review_rows=len(kept),
        assistant_reviewed_train_validation_rows=118,assistant_held_rows=118-len(kept),
        assistant_sample_rows=0,topical_counts=None,
        validation_representative=False,training_ready=False,training_started=False,
        human_review_completed=False,reserved_test_status='Opaque inherited test candidates; not human reviewed or acceptance-ready.')
    result['files']={p.name:checksum(p) for p in output.iterdir() if p.is_file()}
    save(output/'manifest.json',result)
    print(json.dumps({'counts':dict(counts),'tokens':dict(tokens),'review_rows':len(kept)},ensure_ascii=False))


if __name__=='__main__':
    build()
