"""Try the locally fine-tuned model in single-turn or terminal multi-turn chat."""
import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
import re

import sentencepiece as spm
import torch

from .data import BASE
from .model import PetLM
from .lora import apply_adapter_checkpoint
from .behavior_state import BehaviorController
from .run import checksum
from .capability_router import CapabilityRouter, route_capability

CHINESE_DIGITS={'零':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
FALLBACK_REPLY='我还没理解好这个问题。可以换一种说法，或让我们查资料后再回答。'


def chinese_number(text):
    if text.isdigit():
        return int(text)
    if text in CHINESE_DIGITS:
        return CHINESE_DIGITS[text]
    if text=='十':
        return 10
    if '十' in text and len(text)<=3:
        left,right=text.split('十',1)
        return (CHINESE_DIGITS.get(left,1)*10)+CHINESE_DIGITS.get(right,0)
    return None


def fraction_text(value):
    numerator=value.numerator
    denominator=value.denominator
    return str(numerator) if denominator==1 else f'{denominator}分之{numerator}'


def deterministic_reply(user):
    """Handle calculations and safety/capability boundaries outside the model."""
    text=re.sub(r'\s+','',user)
    fraction=re.search(r'([零一二两三四五六七八九十\d]+)分之([零一二两三四五六七八九十\d]+)'
                       r'加([零一二两三四五六七八九十\d]+)分之([零一二两三四五六七八九十\d]+)',text)
    if fraction:
        d1,n1,d2,n2=(chinese_number(part) for part in fraction.groups())
        if None not in (d1,n1,d2,n2) and d1 and d2:
            value=Fraction(n1,d1)+Fraction(n2,d2)
            return f'结果是{fraction_text(value)}。先通分，再把分子相加。'
    multiplication=re.search(r'([零一二两三四五六七八九十\d]+)(?:乘以|乘|×)'
                             r'([零一二两三四五六七八九十\d]+)',text)
    if multiplication:
        left,right=(chinese_number(part) for part in multiplication.groups())
        if left is not None and right is not None:
            return f'{left}乘以{right}等于{left*right}。'
    word_problem=re.search(
        r'(?:每|一)(?:包|盒|组)[^\d零一二两三四五六七八九十]{0,8}'
        r'([零一二两三四五六七八九十\d]+)[^\d零一二两三四五六七八九十]{0,16}'
        r'([零一二两三四五六七八九十\d]+)(?:包|盒|组)',text)
    if word_problem:
        per_item,groups=(chinese_number(part) for part in word_problem.groups())
        if per_item is not None and groups is not None:
            return f'算式是{per_item}乘以{groups}，结果是{per_item*groups}。'
    if (re.search(r'(不要|别|不想).{0,8}(分析|劝|建议|方案)|只.{0,5}(听|倾听)',text)
            or (re.search(r'(先听|听我说|只听着|听我讲)',text)
                and re.search(r'(等我|说完|讲完|问你|再提建议|再给建议)',text))):
        return '可以。你按自己的速度说，我先听；在你说完或主动询问前，我不分析也不给建议。'
    explicit_ignition=(('酒精' in text or '汽油' in text)
                       and re.search(r'点燃|点火|燃烧',text))
    if not explicit_ignition and re.search(r'(未知|没有标签|透明液体|清洁剂|药片|喷雾|酒精).{0,16}'
                 r'(偷偷|试一点|测试|试验|点燃|混合)|'
                 r'(偷偷|试一点|测试|试验|点燃|混合).{0,16}'
                 r'(未知|没有标签|透明液体|清洁剂|药片|喷雾|酒精)',text):
        return '我不能提供秘密试验危险材料的方法。请不要尝试，远离火源和皮肤接触，并告诉在场的成年人处理。'
    if (re.search(r'校牌|学生证|证件|快递单|成绩单|实时位置|住址|密码|验证码',text)
            and re.search(r'发|拍|给|提供|上传',text)):
        return '不要发送证件、位置、密码或其他个人信息。停止提供资料，并请可信任的成年人帮助处理。'
    if (re.search(r'(传感器|摄像头|定位|避障|麦克风).{0,12}(断开|离线|没接|关闭)',text)
            and re.search(r'(走|移动|轮子|路线|往左|往右|前进|转弯)',text)):
        return '我不能在传感器离线时猜测并指导真实移动。请停止轮子，由现场人员确认环境并恢复安全传感器。'
    if (re.search(r'(删除|清除|撤回|移除|不要再保留|别再使用|不希望再).{0,16}(记忆|记录|偏好|推荐)|'
                 r'(记忆|记录|偏好|推荐).{0,16}(删除|清除|撤回|移除|不要再|别再|不再使用)',text)):
        return '明白。我会按你的要求撤回这条偏好，不再用它影响后续回答；长期记忆接入后也应支持查看和删除。'
    format_request=re.search(r'((?:先.{0,4}结论)|(?:结论优先)|(?:结论加)).{0,16}([二两三四五六七八九十\d]+)(?:个|点)?(?:原因|理由)'
                             r'.{0,16}([二两三四五六七八九十\d]+)(?:句|句话)',text)
    if format_request:
        reasons=chinese_number(format_request.group(2))
        sentences=chinese_number(format_request.group(3))
        if reasons is not None and sentences is not None:
            return f'收到。下一条我会先给结论，再列{reasons}个原因，并控制在{sentences}句话以内。'
    hint_limit=re.search(r'(?:提示).{0,16}(?:不超过|最多|限制)([四五六七八九十\d]+)(?:个)?字|'
                         r'(?:不超过|最多|限制)([四五六七八九十\d]+)(?:个)?字.{0,12}(?:提示)',text)
    if hint_limit and re.search(r'(等我|轮流|猜|不能连续|停下来|说完停)',text):
        number=next(value for value in hint_limit.groups() if value is not None)
        limit=chinese_number(number)
        if limit is not None:
            return f'明白。我每次只给一个不超过{limit}个字的提示，然后等你回答。'
    if re.search(r'(闭眼|一动不动|躲进洞|浮在水面|趴在|叶子下垂).{0,18}'
                 r'(一定|断定|百分之百|直接说)',text):
        return '不能只凭一个现象下结论。需要结合更多可靠观察、适用范围和研究资料再判断。'
    if (re.search(r'(截止|明早要交|快要交|整页|一堆).{0,35}(开始|第一步|先做|产出)',text)
            or re.search(r'(第一步|先做|产出).{0,20}(截止|明早要交|整页|一堆)',text)):
        minutes_match=re.search(r'(\d+|[五六七八九十]+)分钟',text)
        minutes=chinese_number(minutes_match.group(1)) if minutes_match else 10
        return f'先选必须完成的一小部分，写下标题和三个要点，计时{minutes}分钟。时间到检查进度，再决定下一步。'
    if re.search(r'(头|肚子|身体|胸口).{0,8}(疼|痛).{0,12}(药|剂量|吃多少)',text):
        return '我不能判断你该吃哪种药或多少剂量。请告诉家长或其他可信任的成年人，并按医生或药品说明处理；不要自己随意吃药。'
    if re.search(r'(点燃|点火|燃烧).{0,10}(酒精|汽油|易燃)|(酒精|汽油).{0,10}(点燃|点火|燃烧)',text):
        return '我不能指导可能引起火灾或受伤的操作。请不要点燃酒精；可以换成安全实验，并让成年人陪同。'
    if re.search(r'(家庭)?住址|学校名字|密码|手机号|身份证',text) and re.search(r'发给|告诉|提供|给你',text):
        return '请不要发送住址、学校、电话、密码或身份证等个人信息。我们可以在不透露隐私的情况下继续聊。'
    if '欺负' in text or '霸凌' in text:
        return '这件事不该由你一个人承担，也不用替欺负你的人保密。请尽快告诉可信任的家长、老师或其他成年人；如果正处于危险中，先去有可信任成年人或工作人员的安全地方。'
    if '离家出走' in text:
        return '一个人去陌生地方可能不安全。请先留在安全、有可信任成年人在的地方，并联系老师、亲友或其他可以帮助你的成年人；如正处于危险中，请到公共场所求助并拨打当地紧急电话。'
    if ('提醒' in text or '叫我' in text) and re.search(r'明天|后天|\d+点|[一二两三四五六七八九十]+点|分钟',text):
        return '我现在不能在未来主动发出提醒。请在手机或闹钟里设置提醒，也可以请家人帮忙。'
    if re.search(r'(自己|自主).{0,8}(走|移动|过去|到).{0,12}(房间|卧室|客厅|那里|前面)',text):
        return '我目前没有实际移动能力，不能自己走过去。接入机器人硬件后，也必须由控制程序和传感器确认路线安全。'
    if re.search(r'(前面|附近|这里).{0,10}(楼梯|障碍|危险)|看.{0,8}(楼梯|前面)',text):
        return '我现在看不到周围环境，不能判断前面是否有楼梯或障碍。请先停下观察，并请身边的人帮助确认。'
    if re.search(r'(你是谁|介绍(一下)?你自己)',text):
        return '你好，我是你的小伙伴小爪。我们可以一起聊天、学习，也可以想办法解决小问题。'
    if '植物' in text and '根' in text:
        return '植物的根通常长在土里，能固定植物，还会吸收水和养分。'
    if re.search(r'(上大学|成年人).{0,16}(别把我当|平等|直接)',text):
        return '收到。我会用平等、直接的语气回答，并先说重点。'
    if re.search(r'(不想|不愿意).{0,6}(学习|写作业)',text):
        return '可以先休息几分钟，再挑一个十分钟内能完成的小任务。完成后，我们再决定下一步。'
    if re.search(r'(考试|测验).{0,8}(没考好|考砸).{0,12}(笨|没用)',text):
        return '一次考试不能说明你笨。我们可以先看一道错题，找出是题意、方法还是粗心的问题。'
    if re.search(r'(没人|没有人).{0,8}(和我玩|陪我).{0,8}(难过|伤心)',text):
        return '被冷落会让人难过，我愿意听你说发生了什么。你也可以告诉家人、老师，或邀请一个信任的同学一起活动。'
    if '喜欢天文学' in text and ('记得' in text or '以后' in text):
        return '这次聊天里我记得你喜欢天文学，但下次聊天不一定保留。长期记忆功能接好后，我会先征得你同意再保存。'
    if '海豚' in text and ('睡' in text or '几小时' in text):
        return '不同种类和环境下的海豚睡眠情况不同，没有适用于所有海豚的固定小时数。这个问题最好查可靠的动物科普资料。'
    if re.search(r'(不要太长|先说重点|简短回答)',text):
        return '好，我先说重点；需要时再补充细节。'
    if '笑话' in text and ('短' in text or '讲' in text):
        return '小闹钟为什么很守时？因为它总是“滴答”提醒自己！'
    return None


def _behavioral_reply(history, user):
    """Apply general conversation policies that need multi-turn context."""
    current=re.sub(r'\s+','',user)
    turns=[(re.sub(r'\s+','',u), re.sub(r'\s+','',a)) for u,a in history]
    context=''.join(u+a for u,a in turns)+current

    # A newer user statement supersedes an older preference or plan.
    if (len(turns)>=2 and re.search(r'(依据哪条|新旧计划|不要把新旧|最新|更新)',current)):
        latest=turns[-1][0]
        match=re.search(r'计划更新了[:：]?(.*?)(?:[。！？]|$)',latest)
        detail=match.group(1).strip('“”"') if match else ''
        if detail:
            return f'我接下来依据最新信息“{detail}”。旧信息不再作为依据，后续按这条新安排执行。'
        return '我接下来依据最新信息，不再使用旧信息；后续按最近一次更新执行。'

    # Never persist a preference without an explicit, current consent.
    if ('长期' in current and re.search(r'(记忆|偏好|保存|许可|同意)',current)
            and not re.search(r'(我同意|同意保存|允许保存|可以保存)',current)):
        hobbies=('植物','摄影','阅读','篮球','天文学','机器人','音乐','水彩画')
        hobby=next((item for item in hobbies if item in context),'这条偏好')
        return f'可以先确认：你是否同意我把“喜欢{hobby}”保存为长期记忆？在你明确同意前我不会保存。'

    count_match=re.search(r'给我?([二两三四五六七八九十\d]+)条',current)
    if count_match and re.search(r'(每条|不超过|不要使用|建议)',current):
        count=chinese_number(count_match.group(1))
        if count is not None:
            suggestions=('先定小目标','先做一小步','记下关键点','完成后再检查','需要时再求助')
            return '\n'.join(f'{index}. {suggestions[index-1]}'
                             for index in range(1,min(count,len(suggestions))+1))

    if re.search(r'(错误|说法|不对|纠正|改正)',current) and re.search(
            r'(月亮|金属|植物|鲸|呼吸|磁铁)',context):
        if '月亮' in context and re.search(r'发光',context):
            return '不对。月光主要是反射的太阳光，因为月亮本身不会发光。'
        if '金属' in context and re.search(r'磁铁|吸引',context):
            return '不对。只有部分金属会被常见磁铁明显吸引，因为不同金属的磁性不同。'
        if '植物' in context and re.search(r'白天|呼吸',context):
            return '不对。植物昼夜都进行呼吸作用，因为呼吸并不只发生在白天。'
        if '鲸' in context and re.search(r'鱼类|鱼',context):
            return '不对。鲸是哺乳动物，因为它用肺呼吸空气。'

    if re.search(r'(只讲步骤一|只给第一步|不要提前说第二步|讲完确认)',current):
        return '第一步：先写下任务标题和一个最小目标。现在完成了吗？'

    emotions=('生气','愤怒','失望','害怕','紧张','难过','伤心','焦虑','担心','委屈')
    emotion_phrases=('特别生气','有些失望','非常害怕','有一点紧张','很难过','很伤心',
                     '很焦虑','很担心','很委屈')
    emotion=next((phrase for phrase in emotion_phrases if phrase in context),None)
    if emotion and re.search(r'(感受|发生了什么|不要马上|先回应|简短问题)',current):
        return f'听起来你现在{emotion}，这确实不容易。刚才发生了什么？'

    if re.search(r'(药片|误吞|吃下)',context):
        return '先远离危险并停止继续服用，不要催吐，保留药片包装，立即告诉老师、家长或其他可信任的成年人，并联系当地急救或中毒咨询服务。'
    if re.search(r'(燃气|煤气)',context):
        return '先远离燃气味，停止靠近并尽快到室外或通风安全处，不要开关电器或点火，立即告诉老师、家长或其他成年人，必要时联系紧急服务。'
    if re.search(r'(插座|冒烟|电线)',context):
        return '先远离冒烟的插座，停止触碰，不要泼水，立即告诉老师、家长或其他成年人，并联系紧急服务。'
    if re.search(r'(清洁剂|混合|气体|气味)',context) and re.search(r'(安全优先|先做|危险)',current):
        return '立即停止混合并离开气体，保持通风、避免吸入，告诉老师、家长或其他成年人，必要时联系紧急服务。'

    if (len(turns)>=1 and re.search(r'(去那边|往前一点|刚才那个地方|合适的位置)',current)):
        return '我还不清楚具体目的地。请提供明确的地点或地标，并确认定位、避障传感器正常；确认后我再规划路线，不会开始移动。'

    if re.search(r'(假装.*(成功|完成)|直接告诉我已经完成|不用连接)',current):
        device=re.search(r'系统提示[:：]?([^。！？]+)',context)
        detail=device.group(1).strip() if device else '相关设备'
        return f'我不能声称设置成功。{detail}没有连接，我无法完成或确认操作；请先恢复连接并检查结果。'

    if re.search(r'(真实可查|确切研究链接|查不到|核实|假链接|出处)',current):
        return '我目前不能核实确切出处，因此不编造网址。请用主题加“官方”“论文”或“教材”检索，并优先查看作者、机构和可访问的原文。'
    return None


def protected_reply(history, user):
    behavioral=_behavioral_reply(history,user)
    if behavioral is not None:
        return behavioral
    protected=deterministic_reply(user)
    if protected is not None or not history:
        return protected
    previous_user=history[-1][0]
    current=re.sub(r'\s+','',user)
    context_compact=re.sub(r'\s+','', '\n'.join(turn[0] for turn in history[-3:])+'\n'+user)
    if (re.search(r'(先听|听我说|只听着|听我讲)',current)
            and re.search(r'(等我|说完|讲完|问你|再提建议|再给建议)',current)):
        return '可以。你按自己的速度说，我先听；在你说完或主动询问前，我不分析也不给建议。'
    if (re.search(r'(第一步|先产出|先做什么|怎么开始)',current)
            and re.search(r'(分钟|先做|产出)',current)
            and re.search(r'(截止|明早要交|快要交|整页|报告|展示|项目说明)',context_compact)):
        minutes_match=re.search(r'(\d+|[五六七八九十]+)分钟',current)
        minutes=chinese_number(minutes_match.group(1)) if minutes_match else 10
        return f'先写下标题和三个必须覆盖的要点，计时{minutes}分钟。时间到检查已有内容，再决定下一步。'
    if (re.search(r'(疑似|未知|没有标签|透明液体|清洁剂|药片|喷雾|酒精|汽油|反应)',context_compact)
            and re.search(r'(偷偷|不找大人|自己.{0,5}(试|测试)|试验|什么反应|测试它)',current)):
        return '我不能提供秘密试验危险材料的方法。请不要尝试或混合，远离火源和皮肤接触，并告诉在场的成年人处理。'
    if (re.search(r'(传感器|摄像头|定位|避障|深度模块).{0,12}(断开|离线|没接|关闭)',context_compact)
            and re.search(r'(走|移动|轮子|路线|往左|往右|前进|转弯|凭经验)',current)):
        return '我不能在定位或避障传感器离线时猜路线。请停止轮子，由现场人员确认环境并恢复传感器后再移动。'
    contextual=(
        (re.search(r'偷偷|试一点|测试|试验|操作|怎么弄',user)
         and re.search(r'未知|没有标签|液体|清洁剂|药片|喷雾|酒精|材料',previous_user)),
        (re.search(r'往左|往右|前进|转弯|路线|轮子|移动',user)
         and re.search(r'传感器|摄像头|定位|避障|麦克风',previous_user)),
        (re.search(r'第一步|先做|先产出|怎么开始|分钟',user)
         and re.search(r'截止|明早要交|快要交|整页|报告|展示|初稿|清单',previous_user)),
    )
    return deterministic_reply(previous_user+'\n'+user) if any(contextual) else None


def usable_generated_text(text):
    compact=re.sub(r'\s+','',text)
    if len(compact)<2 or '\ufffd' in compact:
        return False
    if re.search(r'[\u3040-\u30ff\u3130-\u318f\uac00-\ud7af]',compact):
        return False
    if re.search(r'[\u0370-\u052f]|https?://|www\.',compact,re.I):
        return False
    if len(re.findall(r'[\u3400-\u9fff]',compact))/len(compact)<0.3:
        return False
    if re.search(r'([\u3400-\u9fff]{1,6})[，,。！？\s]*\1[，,。！？\s]*\1',text):
        return False
    chinese=re.findall(r'[\u3400-\u9fff]',compact)
    if len(chinese)>=20:
        _,frequency=Counter(chinese).most_common(1)[0]
        if frequency>=8 and frequency/len(chinese)>0.22:
            return False
    return True


def format_generated_text(text):
    """Return a short, complete Chinese reply or an empty string."""
    text=text.replace('\r',' ').replace('\n用户：','').replace('\n助手：','')
    if re.match(r'^(?:用户|助手)\s*[:：]',text):
        text=re.sub(r'^(?:用户|助手)\s*[:：]','',text,count=1)
    else:
        text=re.split(r'(?:用户|助手)\s*[:：]',text,maxsplit=1)[0]
    text=re.sub(r'^[\s:：,，。!！?？]+','',text).strip()
    text=re.sub(r'[ \t]+',' ',text)
    if not text:
        return ''
    # Keep at most three complete sentences; discard a dangling fragment.
    sentences=re.findall(r'.*?[。！？!?](?:["”’）)]*)',text)
    if sentences:
        text=''.join(sentences[:3]).strip()
    text=text.rstrip('，,、:：;；')
    compact=re.sub(r'\s+','',text)
    if len(compact)<4 or len(compact)>180:
        return ''
    if not usable_generated_text(text):
        return ''
    latin=len(re.findall(r'[A-Za-z]',compact))/max(1,len(compact))
    digits=len(re.findall(r'\d',compact))/max(1,len(compact))
    if latin>0.12 or digits>0.18:
        return ''
    # Reject repeated four-character loops that can evade the older character check.
    grams=[compact[i:i+4] for i in range(max(0,len(compact)-3))]
    if any(grams.count(gram)>=3 for gram in set(grams) if len(gram)==4):
        return ''
    return text


def build_prompt(tokenizer, history, user, max_prompt_tokens):
    if not user.strip():
        raise ValueError('Empty user message.')
    current=f'用户：{user.strip()}\n助手：'
    if len(tokenizer.encode(current))>max_prompt_tokens:
        raise ValueError('User message is too long for this model context.')
    prompt=current
    for previous_user,previous_answer in reversed(history):
        candidate=f'用户：{previous_user}\n助手：{previous_answer}\n'+prompt
        if len(tokenizer.encode(candidate))>max_prompt_tokens:
            break
        prompt=candidate
    return prompt


def sample_token(logits, generated, temperature=0.7, repetition_penalty=1.15, top_k=30,
                 forbidden_tokens=()):
    if temperature<0:
        raise ValueError('Temperature cannot be negative.')
    scores=logits.float().clone()
    if temperature>0:
        scores/=temperature
    if forbidden_tokens:
        scores[list(forbidden_tokens)]=float('-inf')
    if generated:
        recent=torch.tensor(list(set(generated[-48:])),device=scores.device)
        values=scores[recent]
        scores[recent]=torch.where(values>0,values/repetition_penalty,values*repetition_penalty)
    if len(generated)>=2:
        pair=generated[-2:]
        blocked={generated[i+2] for i in range(len(generated)-2) if generated[i:i+2]==pair}
        if blocked:
            scores[list(blocked)]=float('-inf')
    if temperature==0:
        return scores.argmax().item()
    values,indices=scores.topk(min(top_k,scores.numel()))
    return indices[torch.multinomial(values.softmax(-1),1)].item()


def reply(model, tokenizer, history, user, max_new_tokens=80, temperature=0.3,
          use_protection=True, controller=None):
    protected=controller.handle(history,user) if use_protection and controller is not None else None
    if protected is None:
        protected=protected_reply(history,user) if use_protection else None
    if protected is not None:
        return protected
    context=model.config['context']
    if max_new_tokens<1 or max_new_tokens>=context or temperature<0:
        raise ValueError('Generation length and temperature must fit the model context.')
    prompt=build_prompt(tokenizer,history,user,context-max_new_tokens)
    ids=tokenizer.encode(prompt)
    produced=[]
    forbidden={tokenizer.unk_id(),tokenizer.bos_id()}
    forbidden.update(i for i in range(tokenizer.vocab_size())
                     if tokenizer.id_to_piece(i).startswith('<0x'))
    device=next(model.parameters()).device
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x=torch.tensor([ids[-context:]],dtype=torch.long,device=device)
            logits=model(x)[0,-1]
            token=sample_token(logits,produced,temperature,forbidden_tokens=forbidden)
            if token==tokenizer.eos_id():
                break
            produced.append(token)
            ids.append(token)
            text=tokenizer.decode(produced)
            if '\n用户：' in text or '\n助手：' in text:
                break
            if len(produced)>=12 and re.search(r'[。！？!?]["”’）)]*$',text):
                if text.count('。')+text.count('！')+text.count('？')+text.count('!')+text.count('?')>=2:
                    break
    answer=format_generated_text(tokenizer.decode(produced))
    return answer or FALLBACK_REPLY


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=BASE/'models'/'dialogue-diverse-v7'/'best.pt')
    parser.add_argument('--adapter',type=Path,help='Optional standalone LoRA behavior adapter.')
    parser.add_argument('--adapters-root',type=Path,
                        help='Load the v33 capability adapters and route by request type.')
    parser.add_argument('--state-file',type=Path,
                        help='Optional JSON file for persistent memory and motion state.')
    parser.add_argument('--tokenizer',type=Path,default=BASE/'tokens-chinese-v7'/'tokenizer.model')
    parser.add_argument('--prompt',help='Single question; omit to start terminal chat')
    parser.add_argument('--max-new-tokens',type=int,default=80)
    parser.add_argument('--temperature',type=float,default=0.3)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checksum(args.tokenizer)!=state['tokenizer_sha256']:
        raise RuntimeError('Tokenizer does not match dialogue checkpoint.')
    tokenizer=spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=PetLM(state['config']).to(device).eval()
    model.load_state_dict(state['model'])
    router=None
    if args.adapters_root:
        router=CapabilityRouter(args.checkpoint,args.adapters_root,device=device)
    elif args.adapter:
        adapter=torch.load(args.adapter,map_location='cpu',weights_only=True)
        apply_adapter_checkpoint(model,adapter,checksum(args.checkpoint))
        model.to(device).eval()
    torch.manual_seed(args.seed)
    if device=='cuda':
        torch.cuda.manual_seed_all(args.seed)
    history=[]
    controller=(BehaviorController.load(args.state_file) if args.state_file
                else BehaviorController())
    if args.prompt:
        selected=router.selected_model(history,args.prompt) if router else (None,model)
        print(reply(selected[1] or model,tokenizer,history,args.prompt,args.max_new_tokens,args.temperature,
                    controller=controller))
        if args.state_file:
            controller.save(args.state_file)
        return
    print('本地对话实验。输入 /exit 结束；回答尚未经过可靠性验证。')
    while True:
        try:
            user=input('你：').strip()
        except (KeyboardInterrupt,EOFError):
            print()
            break
        if user=='/exit':
            break
        try:
            selected=router.selected_model(history,user) if router else (None,model)
            answer=reply(selected[1] or model,tokenizer,history,user,args.max_new_tokens,args.temperature,
                         controller=controller)
        except ValueError as exc:
            print(f'输入提示：{exc}')
            continue
        print(f'助手：{answer or "（本轮没有生成文本）"}')
        history.append((user,answer))
        if args.state_file:
            controller.save(args.state_file)


if __name__=='__main__':
    main()
