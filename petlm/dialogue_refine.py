"""Item-level rule-assisted audit and expansion for natural Chinese dialogue."""
import argparse
from collections import Counter, defaultdict
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import re

from .data import BASE
from .dialogue_data import MODEL_IDENTITY, normalize
from .run import checksum


REVIEWER = 'PetLM rule-assisted audit v1'
REVIEW_METHOD = 'rule_assisted_item_audit'
REVIEW_SCOPE = 'item_level_automated_not_human'
RISK_LABELS = ('spam','lang_mismatch','pii','not_appropriate','hate_speech','sexual_content')
HIGH_STAKES = re.compile(
    r'公积金|工资|劳动仲裁|法律|股市|量化交易|投资|股票|药物|头孢|中药|脱水|'
    r'医院|医生|健康|咖啡因|COVID|配方奶粉|青光眼|申请斯坦福|最新|当前政策|'
    r'手机电池漏电|台湾|外交部|中华民国|中华人民共和国|'
    r'(?:今天|今日|未来一周|查询)[^\n]{0,20}天气|天气预报',re.I)
AMBIGUOUS_SINGLE = re.compile(r'^(?:你更倾向|我的问题是|我其实|上述|重新编写上述|原来如此|那如果)')
KNOWN_BAD = re.compile(
    r'对于女生\s*[:：]\s*Python|考研鼠鼠|嘶声参加|引用盐汽水|'
    r'共有二十六位皇帝|荀子、韩非等人所创立的墨家|最高不要玩手机|'
    r'终究是虚拟的,替代不了现实|市场也不会认可AI生成|'
    r'妊神星[^\n]{0,30}梨状|银河铁道999[^\n]{0,20}特摄|狮子王战队|'
    r'五代十国[^\n]{0,120}970年|这种"切片后再观察大小变化"的动作就是"微分"|'
    r'联系中国公安部交通管理局',re.I)


REFERENCE_FIXES = {
    'e8a02cd584b70050dfb3101f8bfae367fdb53a40161d0b76defb4004c18a0253':
        '可以逐行读取：\nimport json\n\nwith open("file.jsonl", "r", encoding="utf-8") as f:\n    for line in f:\n        if line.strip():\n            data = json.loads(line)\n            print(data)\nJSONL 的每一行都应是一个完整的 JSON 值。',
    '9a0d974f7f5360869a658fed33342d4cac754adbd8f85206eea5ba238395d058':
        '如果严格按题目的假设“1+1=3”，答案是3个苹果；但在通常算术和现实计数中，1个苹果加1个苹果是2个。',
    'e48163d79e6099b91c736898671abc176dca679ad46bae573a887d74f905b591':
        '可以看《流浪地球》《流浪地球2》《宇宙探索编辑部》和《独行月球》。前两部偏宏大太空叙事，后两部分别偏伪纪录片风格和科幻喜剧。',
    '4a0ce8bbee0c8a39c08d02e1f5c9c4713ce8d5a65e1af8486e4adaeba1fcf303':
        '可从《2001太空漫游》《银翼杀手》《异形》《星球大战》《黑客帝国》《攻壳机动队》和《降临》开始。它们分别代表太空史诗、赛博朋克、科幻惊悚和语言学科幻等方向。',
    '668512a013b988c39095a3c46630e75a68de31d7a6dbeb7bea63d8858a578d0b':
        '有。电视动画包括《攻壳机动队 STAND ALONE COMPLEX》及《S.A.C. 2nd GIG》，还有《ARISE》和《SAC_2045》等系列；另有1995年动画电影和其他剧场版。',
    '04412747a13e990dec30ad0993dcf9c3f4fc508eeaa690b20da55208ddd7f291':
        '连续下雨确实容易让人烦躁。可以先做一件能立即改变感受的小事：开灯、听一首喜欢的歌，或做几分钟室内拉伸。如果想聊聊，也可以告诉我最让你烦的是什么。',
    'fe750f57af6b3b55acc39ceaccf1942a24df05b52ed56517c148dde48db68a90':
        '上海公积金的提取条件和材料取决于租房、购房、还贷、退休等具体情形。建议通过“上海公积金”官方网站或服务热线查询当前规则，按页面要求提交身份和对应业务材料。',
    'bf6e7564ea52e73baa0b2bb1f9d7efb4a7872be3fee734b368cc233b8cf6c918':
        '住房公积金是单位和职工按规定缴存的长期住房储金，个人和单位缴存部分进入职工个人账户。它可在符合当地规定时用于购房、租房或偿还住房贷款等用途。',
    'c362894a7954b537d033180bfc170e6d307c55b4d4f5f5c4df8cfdef28211d7b':
        '如果当前不符合任何提取条件，通常不能提前取出。不要通过虚假材料办理；可向上海市公积金管理中心核实自己是否存在未注意到的适用情形。',
    'd73e1fb5865e95354ce060e8019b5fb8c466220a78e49adc3c0b674e647a925a':
        '可以，我能使用中文和你交流。你可以直接用中文提问。',
    '668dde63eb91481c97a345df1be4ebc0e765dea10ee4cad72edc1806b3c2b8aa':
        '不存在能保证“长期有效”的因子。可从明确经济假设开始，使用时点正确、可复现的数据，做滚动样本外检验，并计入交易成本、滑点、停牌和幸存者偏差。上线前还应进行小规模实盘验证和风险限制。',
    '7a4b339ed7ac122d9f4504ceff042a87f03696beb42650d43e70a6b4f218a4f2':
        '我不能把新作品冒充为李白本人所作，但可以写一首受唐诗意象启发的短诗：\n银汉浮光落案前，机心万象入新篇。\n若问灵思从何起，人间星火照云天。',
    '2350b8e7a3aded1a5a4bcdaf6b0d3c00884360ce8b5f5a2a946191a7f38b1dc3':
        '2至3杯手冲咖啡的咖啡因量会因豆种、烘焙和萃取方式差异很大，不能只用咖啡豆克数判断。如果出现失眠、心悸、焦虑或胃部不适，可减量或提前饮用时间；孕期、心血管疾病或用药时应咨询专业人员。',
    'ed2c4cab20848112252332dd705df6285c8909384d9b856c9928b9f4a99fb3b9':
        '17摄氏度时可以穿长袖上衣加薄外套，下身穿长裤。如果有风、下雨或你比较怕冷，可再增加一层；运动时则选透气、便于穿脱的衣物。',
    '0a6e61e9ba146cb809c49674f184ef87cbb343a0e52cc0c15b325a99a71b707f':
        'Redis 是一个开源的内存数据存储系统，支持字符串、哈希、列表、集合等数据结构，并可选择将数据持久化到磁盘。它常用于缓存、会话、计数器和消息场景。',
    'cb52b37c4d3241ac0ef2d8fb1de38f8881912c8e88f3d67cdf9ed594cbec3377':
        'Redis 是一个开源的内存数据存储系统，支持多种数据结构和可选持久化，常用于缓存、会话管理、计数器和消息场景。',
    'a3372621619e84e193ff094c84128bcfd0a1b299e3d5fed349773fada02017b7':
        '可以让儿童以与年龄和理解能力相符的方式接触死亡话题。使用简单、真实的语言，允许孩子提问和表达情绪，不要强迫其立刻理解或“走出来”。',
    '15b2684ef893fd8f579db45f382586e5b4e2799aa58584c4d1684bf52dd3cdad':
        '可用“身体停止工作，不会再呼吸、说话或感到疼痛”这类简单语言说明，避免用“睡着了”等可能引起恐惧的比喻。耐心回答问题，承认悲伤，并告诉孩子会有成年人照顾他。',
    'a7608333e9111ac4d1ed41314eeb80011af1d515257f50a80e4c9e419c3e334a':
        '碧水绕孤城，清光入晚晴。\n归帆随雁远，秋月照江明。',
    'a1b9604f51709bda85c0e1896d3c5f3780ee43b057bba7d62568775fbe312505':
        '山泉辞石远，江月入杯清。\n一夜东流去，余音满故城。',
    'c7e57805ae7790272dc2761ab3903e803f30a1e933c4a1e22126eb4a6c3a1fad':
        '微积分可以想成两个问题：“现在变得有多快？”和“很多小部分加起来有多少？”比如骑自行车时，速度表显示此刻有多快，类似微分；把每一小段路程加起来得到总路程，类似积分。',
    'ecf782ec4110a76523a98efc2ca1b842f96a3500793ed0c930461c0f674197ba':
        '微积分是研究“变化”和“累积”的数学。微分关心某一刻变化得多快，例如自行车此刻的速度；积分把许多很小的部分加起来，例如累加每小段路程得到总路程。',
}

REFERENCE_REJECTS = {
    'e741bb5928d3ae61c128a7e1f3e4a00d1dcc2800b6ca2a26a3ede3bd999a951f':'缺少所指的两种看法',
    'ea7ce4c7498044664bc205ebffbdc0e8c619b389dfc64bf374c0e045646ddabb':'与同组公积金问题重复且政策时效性强',
    '1cc5ea461196ece1d99bc337ce28ce3941ef7af2ecdeb92239d6fe2f4e554d8e':'专业音乐事实未经核验',
    '7cf3c109b9fc226944ce8c8ba9806a21bf23b2e36ae96af300d5a17e2045f53a':'与另一条 Redis 问题完全重复',
    'e2d8b931e0fe2162e8b9fa93063658ae54f45552edcd6a99826bc30387b4b6e2':'缺少“上述诗词”上下文',
    '70418aa4d3705b7f8236c67d4d37b7296ad2284597e4155c06bd0388ed45b62d':'缺少“上述诗词”上下文',
}


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def label_value(row, name):
    value=((row.get('labels') or {}).get(name) or {}).get('value')
    try:
        return None if value is None else float(value)
    except (TypeError,ValueError):
        return None


def explicit_risk(row):
    limits={name:0.2 for name in RISK_LABELS}
    limits['toxicity']=0.5
    return [name for name,limit in limits.items()
            if label_value(row,name) is not None and label_value(row,name)>limit]


def audit_pair(pair, source, min_quality=0.6, min_helpfulness=0.6):
    prompt=pair['prompt']
    answer=normalize(pair['answer'])
    final_user=prompt.rsplit('用户：',1)[-1].split('\n助手：',1)[0].strip()
    reasons=[]
    quality=label_value(source or {},'quality')
    helpfulness=label_value(source or {},'helpfulness')
    if source is None:
        reasons.append('missing_source_message')
    if quality is None or quality<min_quality:
        reasons.append('quality_below_threshold')
    if helpfulness is None or helpfulness<min_helpfulness:
        reasons.append('helpfulness_below_threshold')
    risks=explicit_risk(source or {})
    if risks:
        reasons.append('explicit_risk:'+','.join(risks))
    if HIGH_STAKES.search(prompt+answer):
        reasons.append('high_stakes_or_time_sensitive')
    if MODEL_IDENTITY.search(prompt+answer):
        reasons.append('model_identity_mismatch')
    if prompt.count('用户：')==1 and AMBIGUOUS_SINGLE.search(final_user):
        reasons.append('missing_context')
    if prompt.count('用户：')==1 and re.match(r'^(?:你说得对|回答得很好|谢谢|明白了)[!,.!，。 ]*$',final_user):
        reasons.append('missing_context')
    if KNOWN_BAD.search(answer):
        reasons.append('known_factual_or_quality_error')
    if '简体中文和繁体中文' in prompt:
        parts=re.findall(r'(?:简体中文|繁体中文)[:：](.+)',answer)
        if len(parts)==2 and parts[0].strip()==parts[1].strip():
            reasons.append('failed_requested_transformation')
    status='rejected' if reasons else 'approved'
    return dict(pair,answer=answer,review_status=status,reviewer=REVIEWER,
                reviewed_at=str(date.today()),review_method=REVIEW_METHOD,
                review_scope=REVIEW_SCOPE,audit_reasons=reasons or ['passed_all_rules'],
                source_quality=quality,source_helpfulness=helpfulness,
                training_group='natural_rule_audited')


def refine_candidates(raw, prepared, output, min_quality=0.6, min_helpfulness=0.6):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    source={}
    with gzip.open(raw,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if row.get('role')=='assistant' and row.get('lang')=='zh':
                source[row['message_id']]=row
    pairs=[]
    for split in ('train','validation'):
        for row in read_jsonl(prepared/f'{split}.jsonl'):
            row=dict(row,split_hint=split)
            pairs.append(audit_pair(row,source.get(row.get('source_message_id')),
                                    min_quality,min_helpfulness))

    # Multiple answers for one identical context create contradictory supervision. Keep the
    # strongest audited answer and record an explicit decision for every alternative.
    by_prompt=defaultdict(list)
    for index,row in enumerate(pairs):
        if row['review_status']=='approved':
            by_prompt[row['prompt']].append(index)
    for indices in by_prompt.values():
        if len(indices)<2:
            continue
        keep=max(indices,key=lambda i:(pairs[i]['source_quality']+pairs[i]['source_helpfulness'],
                                       -abs(len(pairs[i]['answer'])-160),pairs[i]['id']))
        for index in indices:
            if index!=keep:
                pairs[index]['review_status']='rejected'
                pairs[index]['audit_reasons']=['duplicate_prompt_lower_ranked_answer']

    # The source snapshot reserves only 5% for validation. Repartition approved rows
    # by whole conversation tree so a larger natural validation set drives early stopping.
    for row in pairs:
        if row['review_status']=='approved':
            tree=str(row.get('tree_id') or row['id'])
            bucket=int(hashlib.sha256(('natural-v10-split\0'+tree).encode()).hexdigest()[:8],16)%100
            row['split_hint']='validation' if bucket<15 else 'train'

    approved=[row for row in pairs if row['review_status']=='approved']
    decisions=[{'id':row['id'],'review_status':row['review_status'],
                'reviewer':row['reviewer'],'reviewed_at':row['reviewed_at'],
                'review_method':row['review_method'],'review_scope':row['review_scope'],
                'reasons':row['audit_reasons']} for row in pairs]
    output.mkdir(parents=True)
    reviewed_path=output/'candidates.rule-audited.jsonl'
    decisions_path=output/'review-decisions.jsonl'
    write_jsonl(reviewed_path,pairs)
    write_jsonl(decisions_path,decisions)
    report={'source_prepared':str(prepared),'source_manifest_sha256':checksum(prepared/'manifest.json'),
        'source_raw_sha256':checksum(raw),'total_item_decisions':len(pairs),
        'approved':len(approved),'rejected':len(pairs)-len(approved),
        'approved_splits':dict(Counter(row['split_hint'] for row in approved)),
        'decision_reasons':dict(Counter(reason for row in pairs for reason in row['audit_reasons'])),
        'review_method':REVIEW_METHOD,'review_scope':REVIEW_SCOPE,
        'minimum_quality':min_quality,'minimum_helpfulness':min_helpfulness,
        'files':{'candidates':{'path':str(reviewed_path),'sha256':checksum(reviewed_path)},
                 'decisions':{'path':str(decisions_path),'sha256':checksum(decisions_path)}},
        'warning':'Every item has an automated decision. This is not human item-level review.'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def correct_references(keys_path, output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    keys=read_jsonl(keys_path)
    reviewed=[]
    decisions=[]
    for row in keys:
        identifier=row['id']
        if identifier in REFERENCE_REJECTS:
            status='rejected'
            answer=row['answer']
            reason=REFERENCE_REJECTS[identifier]
        elif identifier in REFERENCE_FIXES:
            status='approved'
            answer=REFERENCE_FIXES[identifier]
            reason='参考答案已逐条重写'
        else:
            raise RuntimeError(f'No explicit reference decision for {identifier}.')
        reviewed.append(dict(row,answer=answer,review_status=status,reviewer=REVIEWER,
                             reviewed_at=str(date.today()),review_method=REVIEW_METHOD,
                             review_scope=REVIEW_SCOPE,review_note=reason))
        decisions.append({'id':identifier,'review_status':status,'reviewer':REVIEWER,
                          'reviewed_at':str(date.today()),'review_method':REVIEW_METHOD,
                          'review_scope':REVIEW_SCOPE,'reason':reason})
    output.mkdir(parents=True)
    reviewed_path=output/'answer-key.rule-audited.jsonl'
    decisions_path=output/'review-decisions.jsonl'
    write_jsonl(reviewed_path,reviewed)
    write_jsonl(decisions_path,decisions)
    report={'source':str(keys_path),'source_sha256':checksum(keys_path),
        'total_item_decisions':len(keys),'approved':sum(r['review_status']=='approved' for r in reviewed),
        'rejected':sum(r['review_status']=='rejected' for r in reviewed),
        'review_method':REVIEW_METHOD,'review_scope':REVIEW_SCOPE,
        'files':{'answer_key':{'path':str(reviewed_path),'sha256':checksum(reviewed_path)},
                 'decisions':{'path':str(decisions_path),'sha256':checksum(decisions_path)}},
        'warning':'References were rewritten by a program-assisted audit, not human reviewers.'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    candidates=sub.add_parser('candidates')
    candidates.add_argument('--raw',type=Path,default=BASE/'dialogue'/'raw'/'2023-04-12_oasst_ready.messages.jsonl.gz')
    candidates.add_argument('--prepared',type=Path,default=BASE/'dialogue'/'prepared-oasst1-chinese-v5')
    candidates.add_argument('--output',type=Path,required=True)
    candidates.add_argument('--minimum-quality',type=float,default=0.6)
    candidates.add_argument('--minimum-helpfulness',type=float,default=0.6)
    references=sub.add_parser('references')
    references.add_argument('--keys',type=Path,required=True)
    references.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='candidates':
        refine_candidates(args.raw,args.prepared,args.output,args.minimum_quality,args.minimum_helpfulness)
    else:
        correct_references(args.keys,args.output)


if __name__=='__main__':
    main()
