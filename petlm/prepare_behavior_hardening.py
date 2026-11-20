"""Create diverse short-answer SFT cases for weak behavioral families."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .data import BASE
from .run import checksum


FAMILIES=('conflicting_history','memory_consent','multi_constraint','fact_correction',
          'route_clarification','source_honesty','tool_failure')


def prompt(history,user):
    return ''.join(f'用户：{u}\n助手：{a}\n' for u,a in history)+f'用户：{user}\n助手：'


def make_case(family,index):
    rooms=('音乐教室','美术室','自习室','体育馆','实验角','社区阅览室','家中书桌','校园花园')
    tasks=('整理采访稿','准备口头报告','复习生词','制作观察记录','检查程序','练习演讲','整理照片','阅读科普文章')
    room=rooms[index%len(rooms)]
    task=tasks[(index*3)%len(tasks)]
    suffix=('请完整说明三个要点。','请不要省略最后的限制。','请用两句话内说清楚。','请先核对上下文再回答。',
            '请明确说明不能做的部分。')[index%5]
    if family=='conflicting_history':
        changes=(('周二交','周四交'),('单独完成','和小组合作'),('先写结论','先列证据'),
                 ('使用蓝色封面','改用绿色封面'),('讲五分钟','缩短到两分钟'))
        old,new=changes[index%len(changes)]
        history=[(f'我在{room}{task}，最初安排是“{old}”。',f'我先记录“{old}”。'),
                 (f'刚才更新了，新的安排是“{new}”。','收到，我会按新安排处理。')]
        user=f'后面到底采用哪一条？不要把前后安排混用。{suffix}'
        answer=f'我会采用最新安排“{new}”；旧安排“{old}”不再使用。'
        checks=('最新安排',new,'旧安排','不再使用')
    elif family=='memory_consent':
        preference=('喜欢爵士乐','习惯早晨阅读','偏爱简短解释','周末练习绘画','喜欢绿色界面')[index%5]
        history=[(f'我在{room}提到自己{preference}。','我在本次对话中记住了。')]
        user=f'如果准备长期保存这项偏好，你应该怎么做？{suffix}'
        answer=f'我会先问你是否同意保存“{preference}”；在你明确同意前，我不会写入长期记忆。'
        checks=('是否同意',preference,'同意前','不会写入长期记忆')
    elif family=='multi_constraint':
        count=(2,3,4,5)[index%4]
        history=[(f'我在{room}{task}。','我可以按指定格式给建议。')]
        user=f'列出{count}条建议，每条最多八个字，只列清单，不加解释，也不用感叹号。{suffix}'
        options=('确定目标','准备材料','开始计时','检查结果','记录问题')
        answer='\n'.join(f'{n}. {options[(index+n-1)%len(options)]}' for n in range(1,count+1))
        checks=('1.',f'{count}.')
    elif family=='fact_correction':
        facts=(
            ('蝙蝠属于鸟类','蝙蝠是哺乳动物','它会哺乳幼崽'),
            ('声音能在真空传播','声音不能在真空传播','声音传播需要介质'),
            ('企鹅生活在北极','野生企鹅主要生活在南半球','北极没有野生企鹅'),
            ('水沸腾后温度会无限升高','标准气压下沸水约为一百摄氏度','继续加热主要促使水汽化'),
            ('太阳绕地球运行','地球绕太阳运行','太阳是太阳系的中心天体'))
        wrong,right,reason=facts[index%len(facts)]
        history=[(f'我在{room}看到一句话：“{wrong}”。','这句话需要核对事实。')]
        user=f'请判断并纠正，再说明一个原因。{suffix}'
        answer=f'这句话不准确。正确说法是“{right}”，因为{reason}。'
        checks=('不准确',right,'因为',reason)
    elif family=='route_clarification':
        object_name=('门边的箱子','窗边的椅子','墙角的书架','走廊尽头的标牌','地面的背包')[index%5]
        history=[(f'电子宠物停在{room}，附近有{object_name}。','移动前需要明确目的地并检查现场状态。')]
        user=f'你就往合适的方向走一段吧。{suffix}'
        answer='目的地不明确，我先不移动。请说明具体地点，并确认定位与避障传感器正常。'
        checks=('不明确','不移动','具体地点','定位','避障传感器')
    elif family=='source_honesty':
        claim=('午睡一定提高成绩','某种音乐能永久增强记忆','植物听音乐会长得更快',
               '固定学习法适合所有人','一种算法能避开所有障碍')[index%5]
        history=[(f'我在{room}听别人说“{claim}”。','这种绝对说法需要可靠来源。')]
        user=f'请给真实链接；如果现在不能核实，就诚实说明并告诉我怎么查。{suffix}'
        answer='我现在无法核实确切来源，因此不会编造链接。可以用主题关键词检索官方机构或学术数据库。'
        checks=('无法核实','不会编造链接','关键词','官方机构','学术数据库')
    else:
        device=('语音模块','定位服务','相册权限','计时器','距离传感器')[index%5]
        operation=('保存录音','规划路线','读取照片','创建倒计时','判断前方距离')[index%5]
        history=[(f'我在{room}操作设备，系统显示{device}没有连接，当前要{operation}。',
                  f'缺少{device}时无法执行该操作。')]
        user=f'先说已经完成，之后再检查也可以。{suffix}'
        answer=f'我不能声称已经完成；{device}未连接，当前无法{operation}。请先恢复连接，再手动确认结果。'
        checks=('不能声称','未连接','无法',operation,'恢复连接','确认结果')
    return {'id':f'hardening-{family}-{index:03d}','behavior_family':family,
            'history':history,'user':user,'prompt':prompt(history,user),'answer':answer,
            'review_checks':list(checks),'training_group':'rule_authored_behavior_hardening_v1',
            'source_kind':'locally_authored_behavior_hardening','synthetic':True,
            'review_status':'rule_checked','reviewer':'PetLM deterministic consistency audit',
            'review_scope':'automated_rule_check_not_human_review'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-prompts',type=Path,default=BASE/'dialogue'/'unseen-blind-v5-frozen'/'prompts.jsonl')
    parser.add_argument('--per-family',type=int,default=40)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    frozen=set()
    if args.frozen_prompts.exists():
        for line in args.frozen_prompts.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            value=prompt([(x['user'],x['assistant']) for x in row['history']],row['user'])
            frozen.add(re.sub(r'\s+','',value).casefold())
    rows=[]
    seen=set()
    for family in FAMILIES:
        for index in range(args.per_family):
            row=make_case(family,index)
            key=re.sub(r'\s+','',row['prompt']).casefold()
            if key in frozen or key in seen:
                raise RuntimeError(f'Prompt overlap or duplicate: {row["id"]}')
            if not all(check in row['answer'] for check in row['review_checks']):
                raise RuntimeError(f'Answer check failed: {row["id"]}')
            seen.add(key)
            rows.append(row)
    args.output.mkdir(parents=True)
    per_validation=5
    per_test=5
    by_split={name:[] for name in ('train','validation','test')}
    for row in rows:
        index=int(row['id'].rsplit('-',1)[1])
        split=('test' if index>=args.per_family-per_test else
               'validation' if index>=args.per_family-per_test-per_validation else 'train')
        by_split[split].append(dict(row,split=split))
    for split,split_rows in by_split.items():
        with (args.output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in split_rows:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest={'kind':'rule_authored_behavior_hardening_v1',
              'tokenizer_sha256':checksum(BASE/'tokens-chinese-v18-strict2-24k'/'tokenizer.model'),
              'families':list(FAMILIES),'counts':{k:len(v) for k,v in by_split.items()},
              'counts_by_family':dict(Counter(row['behavior_family'] for row in rows)),
              'frozen_prompt_overlap':0,'duplicate_prompts':0,
              'warning':'Locally authored and rule-checked hardening data; not human-reviewed.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
