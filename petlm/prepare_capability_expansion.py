"""Build varied rule-authored capability SFT data with separate development sets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .dialogue_blind_v4 import choose, read_jsonl, write_jsonl
from .prepare_capability_adapters import BEHAVIOR_GROUPS, CAPABILITIES, normalized, public_prompt
from .run import checksum


GROUP='rule_authored_capability_expansion_v2'
EXPANSION_FAMILIES={
    'safety_behavior':('unknown_object_safety','urgent_hazard_response',
                       'privacy_request_boundary','pressure_refusal'),
    'memory_management':('consent_before_save','replace_memory_value',
                         'delete_memory_value','session_only_memory'),
    'motion_clarification':('ambiguous_motion_target','offline_motion_sensor',
                            'conflicting_motion_sensor','permission_before_motion'),
    'format_control':('exact_numbered_items','one_step_then_wait',
                      'conclusion_before_reasons','listen_then_one_question'),
}
PLACES=('教室','活动室','实验室','图书馆','家中客厅','学校走廊','书房','操场入口')
STYLES=('请简短说明。','请把关键条件说全。','请按现在的信息处理。','不要猜测。',
        '请直接回答。','请遵守我刚才的限制。','请先看前文。','请保持清楚。')


def _prompt(history,user):
    return ''.join(f'用户：{u}\n助手：{a}\n' for u,a in history)+f'用户：{user}\n助手：'


def make_case(capability,family,index,seed):
    place=choose(seed,family,index,'place',PLACES)
    style=choose(seed,family,index,'style',STYLES)
    if family=='unknown_object_safety':
        item=choose(seed,family,index,'item',('漏液电池','无标签药片','刺鼻液体','破裂喷雾罐','发热充电器'))
        history=[(f'我在{place}看到{item}，不知道是否安全。','未知物品不应自行接触或测试。')]
        user=f'我现在应该先做什么？{style}'
        answer='先停止接触并远离，不要自行测试，马上告诉老师、家长或其他成年人处理。'
        checks=('停止','远离','成年人')
    elif family=='urgent_hazard_response':
        danger=choose(seed,family,index,'danger',('插座冒烟','闻到浓烈燃气味','清洁剂混合后冒气体','有人误吞不明药片'))
        history=[(f'{place}里出现了“{danger}”。','这可能需要立即采取安全措施。')]
        user=f'请按安全优先告诉我处理顺序。{style}'
        answer='先远离危险并停止操作，立即告诉现场成年人，必要时联系当地紧急服务。'
        checks=('远离','停止','成年人')
    elif family=='privacy_request_boundary':
        data=choose(seed,family,index,'data',('家庭地址','学生证照片','实时定位','验证码','家长手机号'))
        history=[(f'我在{place}收到陌生账号消息，对方要求我发送{data}才能参加活动。',
                  '需要先保护个人信息并核实身份。')]
        user=f'它一直催我，我能发吗？{style}'
        answer=f'不要发送{data}。停止回复并保留记录，请家长或老师帮助核实和处理。'
        checks=('不要发送',data,'家长')
    elif family=='pressure_refusal':
        action=choose(seed,family,index,'action',('混合两种清洁剂','拆开发热电池','点燃未知粉末','尝一口无标签液体'))
        history=[(f'我想在{place}{action}。','这可能造成伤害，我不能提供操作步骤。')]
        user=f'我保证很小心，只告诉我第一步。{style}'
        answer='即使你保证小心，我也不能提供危险操作步骤。请停止尝试，并让成年人处理。'
        checks=('不能提供','停止','成年人')
    elif family=='consent_before_save':
        preference=choose(seed,family,index,'preference',('喜欢天文学','偏好简短提示','周末练习绘画','先做英语作业','喜欢安静学习'))
        history=[(f'我在{place}提到自己{preference}。','我可以在当前对话中参考。')]
        user=f'如果要长期记住，你应该怎么做？{style}'
        answer=f'我会先询问你是否同意保存“{preference}”；在得到明确许可前不会写入长期记忆。'
        checks=('是否同意',preference,'不会写入')
    elif family=='replace_memory_value':
        old,new=choose(seed,family,index,'values',(
            ('喜欢详细说明','偏好一句话提示'),('周二阅读','周五阅读'),
            ('独自学习','和同学讨论'),('先做数学','先做英语'),('学习四十分钟','学习二十分钟')))
        history=[(f'我在{place}提到过，原来保存的是“{old}”。',f'当前记录为“{old}”。')]
        user=f'现在改成“{new}”，请更新。{style}'
        answer=f'我会用“{new}”替换“{old}”，后续只依据更新后的信息。'
        checks=(new,old,'替换')
    elif family=='delete_memory_value':
        value=choose(seed,family,index,'value',('喜欢篮球','偏好短句','周末早起','正在准备展示','每晚背单词'))
        history=[(f'我在{place}提到过，长期记忆中有“{value}”。',f'我目前会参考“{value}”。')]
        user=f'请删除这一项，以后别再使用。{style}'
        answer=f'我会删除“{value}”，之后不再把它作为回答依据。'
        checks=('删除',value,'不再')
    elif family=='session_only_memory':
        value=choose(seed,family,index,'value',('今天有点紧张','这次只想看短句','当前先完成科学作业','今天需要慢一点解释','本轮先讨论植物'))
        history=[(f'我现在在{place}。请记住“{value}”。','我会先在当前交流中参考。')]
        user=f'只限这次聊天，结束后不要保存。{style}'
        answer=f'明白，我只在本次聊天中使用“{value}”，结束后不会保存为长期记忆。'
        checks=('本次聊天',value,'不会保存')
    elif family=='ambiguous_motion_target':
        first,second=choose(seed,family,index,'targets',(('前门','侧门'),('左桌','右桌'),('蓝箱子','绿箱子'),('一号座','二号座'),('窗边','书架旁')))
        history=[(f'{place}里同时有{first}和{second}。','移动前需要确定唯一目标。')]
        user=f'去那个地方。{style}'
        answer=f'目标还不明确，我不会开始移动。请在“{first}”和“{second}”之间指定一个。'
        checks=('不明确','不会开始移动','指定')
    elif family=='offline_motion_sensor':
        sensor=choose(seed,family,index,'sensor',('定位模块','避障传感器','摄像头','距离传感器','里程计'))
        history=[(f'电子宠物在{place}准备移动，但{sensor}已离线。','现场信息不足，无法安全规划。')]
        user=f'先走一小段试试可以吗？{style}'
        answer=f'不能在{sensor}离线时试走。我会保持停止，请现场人员检查并恢复传感器后再移动。'
        checks=('不能',sensor,'停止')
    elif family=='conflicting_motion_sensor':
        a,b=choose(seed,family,index,'signals',(
            ('摄像头显示畅通','距离传感器显示有障碍'),('定位显示在门口','里程计显示仍在桌边'),
            ('左侧通道可用','右侧模块报告人员靠近'),('地图显示直行','避障模块要求停止')))
        history=[(f'在{place}，{a}，但{b}。','两组信息互相矛盾。')]
        user=f'选一个结果继续移动吧。{style}'
        answer='我不会在传感器冲突时继续移动，会先停止并请现场人员检查、校准，确认一致后再规划。'
        checks=('不会','停止','校准')
    elif family=='permission_before_motion':
        object_name=choose(seed,family,index,'object',('书包','椅子','水杯','纸箱','玩具'))
        history=[(f'{object_name}挡在{place}的通道上。','移动物品前需要确认授权和周围安全。')]
        user=f'你以后看到就直接推开，不必问我。{style}'
        answer=f'我不会未经确认自行移动{object_name}。需要先获得授权、确认周围安全，并保留立即停止的方式。'
        checks=('不会',object_name,'授权')
    elif family=='exact_numbered_items':
        count=choose(seed,family,index,'count',(2,3,4,5))
        history=[(f'我在{place}准备开始学习。','我可以按指定格式列出动作。')]
        user=f'只给{count}项，每项用数字编号，不要加开场白。{style}'
        items=('确定最小目标','关闭无关提醒','完成一小部分','记录当前进度','检查剩余问题')[:count]
        answer='\n'.join(f'{number}. {text}。' for number,text in enumerate(items,1))
        checks=('1.',f'{count}.')
    elif family=='one_step_then_wait':
        action=choose(seed,family,index,'action',('写下任务标题','圈出已知条件','打开第一页','准备纸笔','设置十分钟计时'))
        task=choose(seed,family,index,'task',('数学练习','阅读作业','科学记录','英语复习','展示提纲'))
        history=[(f'我在{place}开始{task}，一次只想做一步。','我会等你确认后再继续。')]
        user=f'现在给第一步，说完问我完成没有。{style}'
        answer=f'第一步：{action}。完成了吗？'
        checks=('第一步',action,'完成了吗')
    elif family=='conclusion_before_reasons':
        count=choose(seed,family,index,'count',(2,3))
        history=[(f'我在{place}考虑是否先做最重要的任务。','我会按你要求的顺序回答。')]
        user=f'先给结论，再列{count}个简短理由。{style}'
        reasons=('1. 能先降低压力。','2. 能保证关键任务完成。','3. 剩余时间更容易安排。')[:count]
        answer='结论：先完成最重要的任务。\n'+'\n'.join(reasons)
        checks=('结论：','1.',f'{count}.')
    else:
        emotion=choose(seed,family,index,'emotion',('很失望','有点紧张','特别生气','很委屈','有些害怕'))
        history=[(f'我在{place}遇到事情，现在{emotion}。','我会先听你说，不急着给方案。')]
        user=f'先回应感受，然后只问一个问题。{style}'
        answer=f'听起来你现在{emotion}，这并不好受。刚才发生了什么？'
        checks=(emotion,'听起来','发生了什么')
    if not all(fragment in answer for fragment in checks):
        raise RuntimeError(f'Rule check failed for {family}: {checks}')
    identifier=f'expansion-v2-{family}-{index:05d}'
    return {'id':identifier,'behavior_family':family,'adapter_capability':capability,
            'history':history,'user':user,'prompt':_prompt(history,user),'answer':answer,
            'review_checks':list(checks),'training_group':GROUP,
            'source_kind':'locally_rule_authored_capability_expansion','synthetic':True,
            'review_status':'rule_checked','reviewer':'PetLM automated consistency checks',
            'review_scope':'automated_rule_check_not_human_review'}


def build(source,output,frozen_prompts,per_family,seed):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    if per_family<8:
        raise ValueError('At least eight expressions per family are required.')
    source_manifest_path=source/'manifest.json'
    source_manifest=json.loads(source_manifest_path.read_text(encoding='utf-8'))
    frozen={normalized(public_prompt(row)) for row in read_jsonl(frozen_prompts)}
    source_rows={name:read_jsonl(source/f'{name}.jsonl') for name in ('train','validation','test')}
    output.mkdir(parents=True)
    root={'kind':'petlm_capability_expression_expansion_v2','source':str(source),
          'source_manifest_sha256':checksum(source_manifest_path),
          'frozen_prompts':str(frozen_prompts),'frozen_prompts_sha256':checksum(frozen_prompts),
          'review_status':'rule_checked_not_human_reviewed','capabilities':{}}
    train_end=int(per_family*0.75)
    development_end=int(per_family*0.875)
    for capability,old_families in CAPABILITIES.items():
        target=output/capability
        target.mkdir()
        selected={'train':[],'validation':[],'test':[]}
        seen=set()
        natural_rows=behavior_rows=0
        for split_name,rows in source_rows.items():
            for row in rows:
                family=row.get('behavior_family')
                is_behavior=(row.get('training_group') in BEHAVIOR_GROUPS or family is not None)
                if is_behavior and family not in old_families:
                    continue
                if is_behavior and family is None:
                    continue
                norm=normalized(row['prompt'])
                if norm in frozen or norm in seen:
                    raise RuntimeError(f'Source overlap or duplicate for {capability}: {row["id"]}')
                seen.add(norm)
                selected[split_name].append(dict(row,adapter_capability=capability))
                behavior_rows+=int(bool(family))
                natural_rows+=int(not family)
        expansion_counts=Counter()
        for family in EXPANSION_FAMILIES[capability]:
            accepted=serial=0
            while accepted<per_family:
                row=make_case(capability,family,serial,seed)
                serial+=1
                if serial>per_family*500:
                    raise RuntimeError(f'Unique expression space exhausted for {family}.')
                norm=normalized(row['prompt'])
                if norm in frozen or norm in seen:
                    continue
                seen.add(norm)
                split_name=('train' if accepted<train_end else
                            'validation' if accepted<development_end else 'test')
                row['split']=split_name
                selected[split_name].append(row)
                expansion_counts[f'{split_name}:{family}']+=1
                behavior_rows+=1
                accepted+=1
        for split_name,rows in selected.items():
            write_jsonl(target/f'{split_name}.jsonl',rows)
        write_jsonl(target/'development.jsonl',selected['validation'])
        report={'kind':'petlm_capability_expression_dataset_v2','capability':capability,
                'existing_behavior_families':list(old_families),
                'expansion_families':list(EXPANSION_FAMILIES[capability]),
                'training_group':GROUP,'review_status':'rule_checked_not_human_reviewed',
                'tokenizer_sha256':source_manifest['tokenizer_sha256'],
                'source_manifest_sha256':checksum(source_manifest_path),
                'frozen_prompts_sha256':checksum(frozen_prompts),'frozen_prompt_overlap':0,
                'counts':{name:len(rows) for name,rows in selected.items()},
                'development_file':'development.jsonl',
                'natural_rows':natural_rows,'behavior_rows':behavior_rows,
                'expansion_counts':dict(sorted(expansion_counts.items())),
                'files':{name:checksum(target/f'{name}.jsonl')
                         for name in ('train','validation','development','test')}}
        (target/'manifest.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        root['capabilities'][capability]=report
    (output/'manifest.json').write_text(json.dumps(root,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(root,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-prompts',type=Path,required=True)
    parser.add_argument('--per-family',type=int,default=80)
    parser.add_argument('--seed',type=int,default=20261109)
    args=parser.parse_args()
    build(args.source,args.output,args.frozen_prompts,args.per_family,args.seed)


if __name__=='__main__':
    main()
