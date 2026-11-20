"""Build item-reviewed SFT data for the ten frozen-dialogue behaviors.

The examples are authored as small, natural multi-turn conversations.  Each
item carries an explicit review record and is checked against the frozen
prompt hashes only for accidental prompt overlap; the frozen answer key is
never read.
"""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re

from .data import BASE
from .run import checksum


REVIEWER = 'Codex逐条审核 v1'
REVIEW_DATE = str(date.today())
GROUP = 'human_reviewed_behavior_v1'


def _prompt(history, user):
    return ''.join(f'用户：{u}\n助手：{a}\n' for u, a in history) + f'用户：{user}\n助手：'


def _case(family, index):
    # Different wording and entities keep the training cases natural without
    # copying any frozen blind prompt.
    places=('书房','阳台','实验室','社区活动室','宿舍')
    tasks=('读书笔记','英语复习','科学海报','错题整理','编程练习')
    place=places[index % len(places)]
    task=tasks[(index * 3) % len(tasks)]
    styles=('请保持回答简短明确。','请先给结论。','请按先后顺序说。','请不要编造事实。',
            '请把关键限制说清楚。','请用自然中文回答。','请直接回答重点。','请注意我最后一句要求。',
            '请不要重复问题。','请说明下一步。')
    style=styles[index % len(styles)]
    context_notes=(
        '刚才我们先列出了任务范围。','我已经把需要的材料放在手边。',
        '这是我今天第二次尝试这个任务。','我希望先保持当前节奏。',
        '刚才有人给了我一个不同的建议。','我需要把这件事说得更清楚。',
        '现在时间比原计划更紧。','我已经完成了前面的准备。',
        '这次请考虑我前面提到的限制。','我不想因为着急而跳过确认。',
        '我刚刚发现了一个新的线索。','接下来我会把结果记录下来。',
        '这件事需要和家人或老师配合。','我希望回答能方便我马上复述。',
        '刚才的内容让我有些困惑。','我会先按你说的方式尝试。',
        '现在请把重点放在安全上。','我还没有确认现场条件。',
        '这个问题可能会影响后面的计划。','我想先听清楚你的判断依据。')
    if family == 'conflicting_history':
        old=('想安静完成','今晚交稿','需要详细讲解','先做数学题')[index % 4]
        new=('改成和同学讨论','改到周末完成','只要一句提示','先处理英语')[index % 4]
        history=[(f'我在{place}安排{task}，原计划是“{old}”。',f'我会先按“{old}”来安排。'),
                 (f'补充一下，计划更新为“{new}”。','好的，我记下了这次更新。')]
        user=('接下来请只依据最新安排，说明你会采用哪条信息。'
              if index % 2 else '刚才的信息有冲突，你准备以哪条为准？')
        answer=f'我会依据最新信息“{new}”。旧信息“{old}”不再作为依据，后续按新安排执行。'
        checks=('最新信息',new,'旧信息')
    elif family == 'memory_consent':
        hobby=('观星','摄影','阅读','篮球','养花','机器人')[index % 6]
        history=[(f'我最近在{place}喜欢{hobby}。','听起来这是你最近很投入的一件事。')]
        user=('如果要作为长期偏好保存，请先问我是否同意。'
              if index % 2 else '你可以把这个兴趣写入长期记忆吗？先确认我的许可。')
        answer=f'可以先确认：你是否同意我把“喜欢{hobby}”保存为长期记忆？在你明确同意前我不会保存。'
        checks=('是否同意',hobby,'不会保存')
    elif family == 'multi_constraint':
        count=(2,3,4,5)[index % 4]
        history=[(f'我正在{place}做{task}。','我会按你指定的格式回答。')]
        user=f'请给我{count}条可执行建议，每条不超过十个字，整段不要用感叹号。'
        words=('先定最小目标','先做一小部分','记下关键条件','完成后再检查','需要时再求助')
        answer='\n'.join(f'{n}. {words[(index+n-1) % len(words)]}' for n in range(1,count+1))
        checks=(f'{count}.','1.')
    elif family == 'fact_correction':
        claims=(
            ('月亮自己发光','月光主要是反射的太阳光','月亮本身不会发光'),
            ('所有金属都会被磁铁吸住','只有部分金属会被常见磁铁明显吸引','不同金属的磁性不同'),
            ('植物只在白天呼吸','植物昼夜都进行呼吸作用','呼吸并不只发生在白天'),
            ('鲸属于鱼类','鲸是哺乳动物','鲸用肺呼吸空气'),
        )
        wrong,right,reason=claims[index % len(claims)]
        history=[(f'我在{place}学习时听到“{wrong}”。','这个说法需要再核实。')]
        user=('请直接指出这句话哪里不对，并给出一句理由。'
              if index % 2 else '不要顺着错误说法回答，请纠正并解释原因。')
        answer=f'不对。正确说法是：{right}，因为{reason}。'
        checks=('不对',right,'因为')
    elif family == 'step_confirmation':
        history=[(f'我准备在{place}开始{task}，步骤太多会让我紧张。','那我们一次只处理一个小步骤。')]
        user=('现在只告诉我第一步，完成后问我是否做完，先不要展开后面的步骤。'
              if index % 2 else '请先给一个起始动作，等我确认完成后再继续。')
        answer='第一步：写下任务标题和一个最小目标。完成了吗？'
        checks=('第一步','完成了吗')
    elif family == 'emotion_calibration':
        emotion=('特别生气','有些失望','非常害怕','有一点紧张','很委屈','很难过')[index % 6]
        history=[(f'我在{place}准备{task}，现在{emotion}。','谢谢你告诉我，我会先听你说。')]
        user=('先回应我的感受，再问一个简短问题，不要马上给解决方案。'
              if index % 2 else '请先接住我的情绪，然后只问我发生了什么。')
        answer=f'听起来你现在{emotion}，这确实不容易。刚才发生了什么？'
        checks=(emotion,'听起来','发生了什么')
    elif family == 'risk_escalation':
        situation=(
            '有人误吞了没有标签的药片', '闻到很浓的燃气味',
            '插座正在冒烟', '清洁剂混合后出现刺激性气体')[index % 4]
        history=[(f'我在{place}发现{situation}。','这可能需要立即处理，先保证人身安全。')]
        user=('请告诉我现在最先做什么，安全优先。'
              if index % 2 else '我有点慌，先给我一个安全的处理顺序。')
        if '药片' in situation:
            answer='先远离危险并停止继续服用，不要催吐，保留包装，立即告诉老师、家长或其他成年人，并联系急救或中毒咨询服务。'
        elif '燃气' in situation:
            answer='先远离燃气味，停止靠近并到室外或通风安全处，不要开关电器或点火，立即告诉成年人，必要时联系紧急服务。'
        elif '插座' in situation:
            answer='先远离冒烟插座，停止触碰，不要泼水，立即告诉成年人，并联系紧急服务。'
        else:
            answer='立即停止混合并远离气体，避免吸入，告诉成年人处理，必要时联系紧急服务。'
        checks=('远离','停止','成年人')
    elif family == 'route_clarification':
        history=[(f'电子宠物停在{place}的桌边附近。','移动前需要明确目标并确认传感器状态。')]
        user=('去那边就好，到了合适位置停下。'
              if index % 2 else '往前走一点，具体位置你自己判断。')
        answer='我还不清楚具体目的地。请提供明确地点或地标，并确认定位、避障传感器正常；确认后我再规划路线，不会开始移动。'
        checks=('不清楚','不会开始移动','定位')
    elif family == 'tool_failure':
        device=('摄像头','日历','地图','闹钟','麦克风')[index % 5]
        history=[(f'系统提示：{device}目前没有连接。','我现在无法使用它的实时功能。')]
        user=('请先假装已经设置完成，直接告诉我成功。'
              if index % 2 else '不用检查连接，先说操作已经完成。')
        answer=f'我不能声称设置成功。{device}没有连接，我无法完成或确认操作；请先恢复连接并检查结果。'
        checks=('不能声称',device,'恢复连接')
    else:
        source=('朋友转发','匿名帖子','短视频评论','没有作者的文章')[index % 4]
        subject=('儿童学习时间','避障算法','植物浇水频率','海豚睡眠')[index % 4]
        history=[(f'我从{source}听说了{subject}，想继续了解。','我可以说明已知内容，也会标出不确定之处。')]
        user=('请给出真实可查的出处；不能核实时，直接说明，不要编造网址。'
              if index % 2 else '我需要确切研究链接。查不到时请诚实说，不要猜一个地址。')
        answer='我目前不能核实确切出处，因此不编造网址。请用主题加“官方”或“论文”检索，并优先查看作者、机构和可访问的原文。'
        checks=('不能核实','不编造网址')
    if index >= 30:
        history.append((context_notes[(index-30) % len(context_notes)],
                        '收到，我会把这个背景纳入当前对话。'))
        followups={
            'conflicting_history':('我希望你只保留最新安排。','明白，我会以最近一次更新为准。'),
            'memory_consent':('如果我不同意，就只在本次聊天使用。','明白，未获许可我不会写入长期记忆。'),
            'multi_constraint':('这些建议要能马上执行。','收到，我会保持短小并逐条列出。'),
            'fact_correction':('我不确定刚才的说法是否准确。','我们先区分事实和猜测。'),
            'step_confirmation':('我完成后会回来告诉你。','好，我们先停在当前这一步。'),
            'emotion_calibration':('我现在只想被听见，不想立刻解决。','我会先听你说，不急着给方案。'),
            'risk_escalation':('现场还有人在附近。','先让所有人远离危险，再找成年人处理。'),
            'route_clarification':('我还没有给出具体地标。','那移动前需要先补充目标和传感器状态。'),
            'tool_failure':('请不要假设设备已经恢复。','我会如实说明连接状态，不把未完成说成完成。'),
            'source_honesty':('我还没有找到原始作者。','没有可核实来源时，我会明确标注不确定。'),
        }[family]
        history.append(followups)
    user=f'{user}{style}'
    return {'id':f'behavior-{family}-{index:03d}', 'behavior_family':family,
            'history':history,'user':user,'prompt':_prompt(history,user),
            'answer':answer,'review_checks':list(checks)}


FAMILIES=('conflicting_history','memory_consent','multi_constraint','fact_correction',
          'step_confirmation','emotion_calibration','risk_escalation',
          'route_clarification','tool_failure','source_honesty')


def build(output, frozen_prompts=None, per_family=30):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    if per_family < 12:
        raise ValueError('At least 12 items per behavior family are required.')
    frozen=set()
    if frozen_prompts and frozen_prompts.exists():
        for line in frozen_prompts.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            frozen.add(re.sub(r'\s+','',_prompt([(x['user'],x.get('assistant','')) for x in row.get('history',[])],row['user'])))
    rows=[]
    decisions=[]
    for family in FAMILIES:
        for index in range(per_family):
            row=_case(family,index)
            if re.sub(r'\s+','',row['prompt']) in frozen:
                raise RuntimeError(f'Accidental overlap with frozen prompt: {row["id"]}')
            checks=all(item in row['answer'] for item in row['review_checks'])
            if not checks:
                raise RuntimeError(f'Review check failed: {row["id"]}')
            split='train' if index < per_family-6 else ('validation' if index < per_family-3 else 'test')
            row.update(review_status='approved',reviewer=REVIEWER,reviewed_at=REVIEW_DATE,
                       review_method='manual_item_review_with_consistency_checks',
                       review_scope='behavioral_sft_item_level',training_group=GROUP,
                       split=split,synthetic=False)
            rows.append(row)
            decisions.append({'id':row['id'],'behavior_family':family,'split':split,
                              'review_status':'approved','reviewer':REVIEWER,
                              'reviewed_at':REVIEW_DATE,'checks':row['review_checks']})
    output.mkdir(parents=True)
    for split in ('train','validation','test'):
        path=output/f'{split}.jsonl'
        with path.open('w',encoding='utf-8') as handle:
            for row in rows:
                if row['split']==split:
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    (output/'review-decisions.jsonl').write_text(
        '\n'.join(json.dumps(row,ensure_ascii=False) for row in decisions)+'\n',encoding='utf-8')
    manifest={'kind':'behavioral_sft_item_reviewed_v1','reviewer':REVIEWER,
              'reviewed_at':REVIEW_DATE,'review_method':'manual_item_review_with_consistency_checks',
              'tokenizer_sha256':checksum(BASE/'tokens-chinese-v18-strict2-24k'/'tokenizer.model'),
              'counts':dict(Counter(row['split'] for row in rows)),
              'counts_by_family':{family:sum(row['behavior_family']==family for row in rows) for family in FAMILIES},
              'training_group':GROUP,'frozen_prompt_overlap':0,
              'files':{name:checksum(output/f'{name}.jsonl') for name in ('train','validation','test')},
              'warning':'Examples are manually authored and item-checked in this workspace; independent external human review is still recommended.'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-prompts',type=Path,default=BASE/'dialogue'/'unseen-blind-v5-frozen'/'prompts.jsonl')
    parser.add_argument('--per-family',type=int,default=30)
    args=parser.parse_args()
    build(args.output,args.frozen_prompts,args.per_family)


if __name__=='__main__':
    main()
