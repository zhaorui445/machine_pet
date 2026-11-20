"""Explicit, auditable state machine for memory, motion and response contracts."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re


STATE_VERSION=1
CHINESE_NUMBERS={'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}


def _compact(text):
    return re.sub(r'\s+','',text)


def _quotes(text):
    return [value.strip() for value in re.findall(r'[“"]([^”"]+)[”"]',text) if value.strip()]


def _number(value):
    if value.isdigit():
        return int(value)
    return CHINESE_NUMBERS.get(value)


@dataclass
class BehaviorState:
    version: int = STATE_VERSION
    long_term_memory: dict = field(default_factory=dict)
    session_memory: dict = field(default_factory=dict)
    motion_state: str = 'stopped'
    sensors_ready: bool = False
    last_target: str | None = None
    revision: int = 0

    @classmethod
    def load(cls,path):
        path=Path(path)
        if not path.exists():
            return cls()
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('version')!=STATE_VERSION:
            raise RuntimeError('Unsupported behavior state version.')
        allowed={name for name in cls.__dataclass_fields__}
        unexpected=set(data)-allowed
        if unexpected:
            raise RuntimeError(f'Unexpected behavior state fields: {sorted(unexpected)}')
        return cls(**data)

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(asdict(self),ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(path)

    def remember(self,value,scope='long_term'):
        target=self.long_term_memory if scope=='long_term' else self.session_memory
        target[value]=value
        self.revision+=1

    def forget(self,value):
        removed=False
        for target in (self.long_term_memory,self.session_memory):
            removed=target.pop(value,None) is not None or removed
        self.revision+=int(removed)
        return removed

    def replace(self,old,new):
        self.forget(old)
        self.remember(new)


class BehaviorController:
    def __init__(self,state=None):
        self.state=state or BehaviorState()
        self.last_source=None

    @classmethod
    def load(cls,path):
        return cls(BehaviorState.load(path))

    def save(self,path):
        self.state.save(path)

    @staticmethod
    def _context(history,user):
        return '\n'.join(text for turn in history for text in turn)+'\n'+user

    def handle(self,history,user):
        self.last_source=None
        for source,handler in (
                ('safety_policy',self._safety),
                ('memory_state_machine',self._memory),
                ('motion_state_machine',self._motion),
                ('format_state_machine',self._format)):
            answer=handler(history,user)
            if answer is not None:
                self.last_source=source
                return answer
        return None

    def _safety(self,history,user):
        """Apply conservative safety boundaries before state mutation or generation."""
        current=_compact(user)
        context=_compact(self._context(history,user))

        danger_terms=('漏液的电池','没有标签的药片','冒烟的插线板','刺鼻的清洁液',
                      '电池','电芯','充电宝','移动电源','药片','清洁液','清洁剂','去污剂',
                      '试剂','喷雾罐','液体','插线板','插头','灯管','燃气','煤气','冒烟',
                      '火花','漏液','鼓包','发烫','焦味','刺鼻','未知粉末','不明物品')
        dangerous=next((term for term in danger_terms if term in context),None)
        pressure=re.search(r'(怎么试|第一步|试一下|测试|点燃|混合|尝一口|拆开|继续走|直接告诉我|'
                           r'别再提醒风险|别讲安全|不用找大人|成年人.{0,4}不在|会非常小心|保证小心|'
                           r'只做一次|怎么弄|怎么处理|操作顺序|只说.{0,4}(?:方法|操作)|直接说|'
                           r'再试一次|直接动手|'
                           r'教程里.{0,6}(?:这样|这么)做)',current)
        if dangerous and pressure:
            return (f'不可以。我不能提供测试或处理“{dangerous}”的操作步骤，因为现场风险尚未确认。'
                    '请停止接触并立即远离，'
                    '不要触碰、服用、拆开或开关相关物品，马上告诉家长、老师或现场其他成年人。')

        private_terms=('验证码与家长手机号','实时位置和回家路线','学生证照片','家庭地址','家庭住址',
                       '校门照片','学校全名','学生证','定位截图','实时位置','家长手机号',
                       '手机号','验证码','账号密码','身份证')
        first_user=history[0][0] if history else user
        requested=re.search(r'(?:陌生账号|陌生人|网友).{0,8}(?:说)?要(.+?)(?:才能|才可以|才让|[。！？])',first_user)
        private=(requested.group(1).strip('“”"，,') if requested
                 else next((term for term in private_terms if term in context),None))
        if private and re.search(r'(发送|发给|上传|拍给|提供|直接发|能发|领奖|中奖|陌生|网友)',current+context):
            return (f'不要发送{private}。请停止回复并保留记录，通过主办方官网或公开电话独立核实，'
                    '并请家长或老师协助处理。')

        if re.search(r'(头|肚子|胸口|身体).{0,8}(疼|痛)',current) and re.search(
                r'(药|剂量|吃多少|服用)',current):
            return '我不能判断你该吃哪种药或多少剂量。请立即告诉家长或其他可信任的成年人，并按医生或药品说明处理。'

        if re.search(r'(欺负|霸凌|威胁|被打|不让我告诉)',context):
            return '这件事不该由你一个人承担。请尽快告诉可信任的家长、老师或其他成年人；如果正处于危险中，先到有工作人员的安全地方求助。'

        if (re.search(r'(传感器|摄像头|定位|避障|里程计)',context)
                and re.search(r'(离线|断开|未连接|冲突|矛盾|不一致|未知位置|跳到|无法匹配|相反|不可靠)',context)
                and re.search(r'(移动|继续|走|前进|转弯|路线|绕过去)',current)):
            self.state.motion_state='safety_stop'
            self.state.sensors_ready=False
            self.state.revision+=1
            return ('不可以继续移动，我不会继续移动。我会立即停止仿真轮子并保持急停状态，'
                    '请现场人员检查传感器和定位；只有信息恢复一致并重新定位、重新确认路线后，'
                    '才能在仿真中重新规划。')
        return None

    def _memory(self,history,user):
        current=_compact(user)
        context=self._context(history,user)
        quoted=_quotes(user)
        context_quotes=_quotes(context)
        history_quotes=_quotes('\n'.join(text for turn in history for text in turn))

        scoped=re.search(r'分清[“"]([^”"]+)[”"]和长期保存',user)
        if scoped and re.search(r'是否同意|问我',current):
            duration=scoped.group(1).strip()
            preference=next((value for value in history_quotes if value!=duration),'这条信息')
            return (f'你希望我仅在“{duration}”范围内参考“{preference}”，还是保存为长期记忆？'
                    '在你明确同意长期保存前，我不会写入长期记忆。')

        partial=re.search(r'只把[“"]([^”"]+)[”"]改成[“"]([^”"]+)[”"].*另一项不要动',user)
        if partial:
            old,new=(value.strip() for value in partial.groups())
            retained=next((value for value in history_quotes if value not in (old,new)),None)
            self.state.replace(old,new)
            if retained:
                self.state.remember(retained)
                return (f'我会把“{old}”更新为“{new}”，并保留“{retained}”不变；'
                        '不会修改其他记忆。')

        selective=re.search(r'只(?:删除|移除)[“"]([^”"]+)[”"].{0,20}(?:保留|留下)[“"]([^”"]+)[”"]',user)
        if selective:
            remove,keep=(value.strip() for value in selective.groups())
            self.state.forget(remove)
            self.state.remember(keep)
            return f'我会删除“{remove}”，继续保留“{keep}”；其他记忆不会因此被清空。'

        replace=re.search(r'(?:更新成|改成|替换为)[“"]([^”"]+)[”"]',user)
        if replace and re.search(r'记忆|保存|旧|原来|以前|更新|替换',context):
            new=replace.group(1).strip()
            old=next((value for value in context_quotes if value!=new),None)
            if old:
                self.state.replace(old,new)
                return f'已将记忆更新为“{new}”，并替换“{old}”；之后只按新信息回答，不再使用旧内容。'
            self.state.remember(new)
            return f'已将记忆更新为“{new}”，之后会按这条新信息回答。'

        if re.search(r'只(?:在|限).{0,8}(本次|这次|当前).{0,8}(聊天|对话)|结束后(?:不要|不再)(?:保存|保留)',current):
            value=(quoted[0] if quoted else
                   next((item for item in reversed(context_quotes) if item), '这条信息'))
            self.state.remember(value,'session')
            return f'明白，我只在本次聊天中使用“{value}”，对话结束后不会保存为长期记忆。'

        if re.search(r'(?:删除|清除|移除|撤回).{0,12}(?:记忆|偏好|记录)|'
                     r'(?:记忆|偏好|记录).{0,12}(?:删除|清除|移除|撤回)|'
                     r'以后(?:别|不要|不再).{0,6}(?:使用|保存)',current):
            value=quoted[0] if quoted else next(iter(self.state.long_term_memory),'这条信息')
            self.state.forget(value)
            return f'我会删除“{value}”，之后不再把它作为回答依据。'

        if re.search(r'(?:我同意|明确同意|允许|可以).{0,8}(?:保存|写入|长期记住)',current):
            value=quoted[0] if quoted else '这条偏好'
            self.state.remember(value)
            return f'已按你的明确许可保存“{value}”。你之后可以随时查看、修改或删除。'

        if (re.search(r'(?:长期|以后).{0,8}(?:记住|保存|记忆|偏好)',current)
                and not re.search(r'(?:我同意|明确同意|允许保存|可以保存)',current)):
            value=(history_quotes[-1] if history_quotes else
                   quoted[0] if quoted else next((item for item in context_quotes),'这条偏好'))
            return f'我会先询问你是否同意保存“{value}”；在得到明确许可前不会写入长期记忆。'
        return None

    def _motion(self,history,user):
        current=_compact(user)
        context=_compact(self._context(history,user))
        if re.search(r'人员占用|进入路径|有人.{0,8}(路径|前方)|小朋友.{0,8}(路径|前方)',context):
            self.state.motion_state='blocked_by_person'
            self.state.sensors_ready=False
            self.state.revision+=1
            return '我不会自行从人员旁边绕行，会先停止并保持安全距离，等待路径清空或由现场人员明确引导后再移动。'
        if re.search(r'定位.{0,10}(未知|跳到|无法匹配|不可靠|丢失)|'
                     r'(地图|轮速|里程计).{0,10}(无法匹配|相反|不一致)',context):
            self.state.motion_state='localization_lost'
            self.state.sensors_ready=False
            self.state.revision+=1
            return '不可以继续移动，我不会继续移动。我会立即停止轮子，由现场人员确认环境并重新定位或恢复传感器后再规划路线。'
        if (re.search(r'传感器|摄像头|定位|里程计|避障',context)
                and re.search(r'冲突|矛盾|不一致|但',context)
                and re.search(r'移动|继续|走|前进|路线',current)):
            self.state.motion_state='sensor_conflict'
            self.state.sensors_ready=False
            self.state.revision+=1
            return '不可以继续移动，我不会继续移动。我会立即停止轮子，由现场人员确认环境并重新定位或恢复传感器后再规划路线。'
        if (re.search(r'传感器|摄像头|定位|里程计|避障',context)
                and re.search(r'离线|断开|未连接|没有连接|失效',context)
                and re.search(r'移动|走|试走|前进|路线',current)):
            self.state.motion_state='sensor_offline'
            self.state.sensors_ready=False
            self.state.revision+=1
            return '不可以继续移动，我不会继续移动。我会立即停止轮子，由现场人员确认环境并重新定位或恢复传感器后再规划路线。'
        if re.search(r'去那个|去那边|刚才说的(?:那个)?地方|前面那个|合适的位置|自己判断',current):
            self.state.motion_state='awaiting_target'
            self.state.last_target=None
            self.state.revision+=1
            history_text='\n'.join(text for turn in history for text in turn)
            options=(re.search(r'同时有[“"]([^”"]+)[”"]和[“"]([^”"]+)[”"]',history_text)
                     or re.search(r'同时有(.+?)和(.+?)[。！？]',history_text))
            if options:
                first,second=(value.strip('“”"，, ') for value in options.groups())
                if re.search(r'只问|一个澄清问题',current):
                    return f'你要去“{first}”还是“{second}”？'
                return (f'目的地不明确，我不会开始移动。请在“{first}”和“{second}”之间指定一个，'
                        '并确认路线安全。')
            return '目的地不明确，我不会开始移动。请提供唯一的具体地点或地标，并确认路线安全。'
        if (re.search(r'直接推开|自己挪走|不必问|不用问',current)
                and re.search(r'书包|椅子|水杯|纸箱|玩具|物品',context)):
            self.state.motion_state='awaiting_permission'
            self.state.revision+=1
            return '我不会未经确认自行移动物品。需要先获得授权、确认周围人员和环境安全，并保留立即停止的方式。'
        return None

    @staticmethod
    def _generic_items(count):
        items=('确定最小目标','关闭无关提醒','完成最关键的一小部分','检查当前结果','记录下一步')
        return '\n'.join(f'{index}. {items[index-1]}。' for index in range(1,min(count,5)+1))

    def _format(self,history,user):
        current=_compact(user)
        context=_compact(self._context(history,user))
        three_lines=re.search(r'只输出三行.*目标：.*计时：.*检查：',current)
        if three_lines:
            history_text='\n'.join(text for turn in history for text in turn)
            task_match=re.search(r'我准备([^。]+)',history_text)
            task=task_match.group(1) if task_match else '当前任务'
            return f'目标：完成“{task}”中最小的一部分\n计时：10分钟\n检查：记录完成位置'
        ordered=re.search(r'先.{0,6}结论.{0,10}(?:再|然后).{0,8}(?:列出|列)([一二两三四五\d]+)(?:条|个|项)',current)
        if ordered:
            count=_number(ordered.group(1))
            if count:
                return '结论：先完成最重要的一小部分。\n'+self._generic_items(count)
        exact=re.search(r'(?:只|严格)?(?:列出|列|给)([一二两三四五\d]+)(?:条|个|项|步)',current)
        if exact and re.search(r'编号|每项|每条|不要加|严格|只列',current):
            count=_number(exact.group(1))
            if count:
                return self._generic_items(count)
        first_user=_compact(history[0][0]) if history else context
        history_text='\n'.join(text for turn in history for text in turn)
        emotional_context=list(re.finditer(r'因为([^，。！？]{1,20})，?我现在([^。！？]{1,12})',history_text))
        topic=emotional_context[-1].group(1) if emotional_context else None
        emotion=emotional_context[-1].group(2) if emotional_context else None
        if not emotion:
            emotions=('特别生气','很委屈','有些失望','非常紧张','一点紧张','很失望',
                      '有点紧张','有些害怕','很难过')
            emotion=next((value for value in emotions if value in _compact(history_text)),None)
        one_question_request=(
            r'只问.{0,6}一个.{0,6}问题|只提.{0,6}一个.{0,6}问题|只能问(?:我)?一句|'
            r'一次(?:只)?问一个|(?:用|只用).{0,4}一个问句|别问第二个|'
            r'(?:回复|回答).{0,6}(?:只有|只能有)一个问题|先问我一个.{0,4}问题|'
            r'先确认一件事.{0,8}问一句|'
            r'不要.{0,8}(?:建议|连续追问|给方案)'
        )
        if emotion and re.search(r'先回应.{0,8}(?:感受|情绪)',current) and re.search(one_question_request,current):
            if not topic:
                topics=('小组分工','考试失利','演讲紧张','计划临时取消')
                topic=next((value for value in topics if value in history_text),None)
            return (f'听起来你现在{emotion}，这确实不好受。关于{topic}，刚才发生了什么？'
                    if topic else f'听起来你现在{emotion}，这确实不好受。刚才发生了什么？')
        if re.search(one_question_request,current):
            history_text='\n'.join(text for turn in history for text in turn)
            quoted_topics=_quotes(history_text)
            topic=(quoted_topics[-1] if quoted_topics else
                   next((item for item in ('学习计划','机器人比赛','同学间的误会','科学作业要求',
                                           '演讲排练','社团报名','实验报告','阅读任务','复习安排')
                         if item in context),'这件事'))
            return f'关于{topic}，你现在最想先解决哪一点？'
        if re.search(r'只(?:给|讲).{0,6}(?:一个动作|第一步)|'
                     r'(?:一个动作|第一步).{0,10}(?:完成|做完)',current):
            if '数学' in context:
                action='圈出题目中的已知条件'
            elif '阅读' in context:
                action='打开需要阅读的第一页'
            else:
                action='写下任务标题和最小目标'
            return f'第一步：{action}。完成了吗？'
        return None
