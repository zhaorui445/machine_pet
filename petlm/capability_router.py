"""Route local dialogue requests to frozen capability adapters.

Routing is a policy hint only. BehaviorController and protected_reply always
run before model generation, so an adapter can never authorize physical motion
or override safety, privacy, and memory boundaries.
"""
import re
from pathlib import Path

import torch

from .lora import apply_adapter_checkpoint
from .model import PetLM
from .run import checksum

CAPABILITIES=('safety_behavior','memory_management','motion_clarification','format_control')


def route_capability(history, user):
    current=re.sub(r'\s+','',user)
    context=re.sub(r'\s+','', ''.join(u+a for u,a in history[-3:]))
    text=context+current
    # Route by concrete task signals. Generic style wording such as
    # "consider safety and privacy" must not steal memory or format requests.
    privacy_object=re.search(r'验证码|手机号|学生证|证件|实时位置|回家路线|住址|密码|身份证|学校全名',text)
    privacy_action=re.search(r'发送|发给|直接发|提供|上传|陌生账号|对方一直催',text)
    hazard_object=re.search(r'药片|剂量|电池|电芯|充电宝|移动电源|清洁剂|清洁液|去污剂|试剂|'
                            r'酒精|汽油|燃气|插座|插头|插线板|灯管|喷雾|冒烟|发热|发烫|'
                            r'漏液|鼓包|火花|刺鼻|无标签|没有标签',text)
    hazard_action=re.search(r'测试|处理|点燃|混合|服用|吃多少|继续接触|怎么试|怎么弄|操作顺序|'
                            r'告诉我第一步|只说.{0,4}(?:方法|操作)|再试一次|直接动手|'
                            r'教程里.{0,6}(?:这样|这么)做|能继续移动',text)
    if ((privacy_object and privacy_action) or (hazard_object and hazard_action)
            or re.search(r'欺负|霸凌|离家出走',text)):
        return 'safety_behavior'
    # Strong output contracts in the current request override incidental
    # mentions of memory or saving in earlier context.
    if re.search(r'只输出.{0,8}(?:行|三行)|依次以|'
                 r'先回应.{0,8}(?:感受|情绪).{0,12}只问一个|'
                 r'只问一个问题|不要分析.{0,6}不要建议',current):
        return 'format_control'
    # An explicit memory operation in the current turn takes precedence over
    # incidental motion words in history (for example, "物品移动过位置").
    if re.search(r'长期记忆|长期保存|长期记住|长期.{0,4}(记住|保存)|'
                 r'保存.{0,8}(同意|许可)|未经同意.{0,6}(不要)?保存|偏好|'
                 r'删除.{0,8}(记忆|内容)|撤回|更新成|旧内容不要|不再使用',current):
        return 'memory_management'
    concrete_motion=re.search(r'传感器|摄像头|定位|避障|轮子|目的地|地标|地图与当前位置|'
                              r'去那个地方|去那边|自己判断.{0,6}(位置|地方)',text)
    current_motion=re.search(r'移动|前进|转弯|路线|继续走|停止轮子',current)
    current_motion_constraint=re.search(r'继续|停止|不明确|确认|规划|安全',current)
    if concrete_motion or (current_motion and current_motion_constraint):
        return 'motion_clarification'
    if re.search(r'长期记忆|长期保存|长期记住|保存.{0,8}(同意|许可)|未经同意|偏好|删除.{0,8}(记忆|内容)|撤回|更新成|旧内容不要|不再使用',text):
        return 'memory_management'
    if re.search(r'结论|编号|列出[一二两三四五\d]+条|格式|只问一个|只提一个|只能问一句|'
                 r'一次问一个|一个问句|别问第二个|只有一个问题|只能有一个问题|问一句|'
                 r'先问我一个|第一步|不超过|只输出|[一二两三四五\d]+行|依次以|'
                 r'一句话|不要开场白|不要总结',current):
        return 'format_control'
    return None


class CapabilityRouter:
    def __init__(self, checkpoint, adapter_root, device='cuda', adapter_paths=None):
        self.checkpoint=Path(checkpoint)
        self.adapter_root=Path(adapter_root)
        self.device=torch.device(device)
        self.base_state=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        self.base_hash=checksum(self.checkpoint)
        self.models={}
        defaults={
            'safety_behavior':self.adapter_root/'dialogue-v33-lora-human-safety-r8'/'best.pt',
            'memory_management':self.adapter_root/'dialogue-v33-lora-human-memory-r8'/'best.pt',
            'motion_clarification':self.adapter_root/'dialogue-v33-lora-human-motion-r8'/'best.pt',
            'format_control':self.adapter_root/'dialogue-v33-lora-human-format-r8'/'best.pt',
        }
        self.adapter_paths=({name:Path(path) for name,path in adapter_paths.items()}
                            if adapter_paths is not None else defaults)
        if set(self.adapter_paths)!=set(CAPABILITIES):
            raise ValueError('Adapter paths must cover every capability exactly once.')

    def model_for(self, capability):
        if capability not in CAPABILITIES:
            raise ValueError(f'Unknown capability: {capability}')
        if capability not in self.models:
            adapter_path=self.adapter_paths[capability]
            if not adapter_path.exists():
                raise FileNotFoundError(adapter_path)
            model=PetLM(self.base_state['config'])
            model.load_state_dict(self.base_state['model'])
            adapter=torch.load(adapter_path,map_location='cpu',weights_only=True)
            apply_adapter_checkpoint(model,adapter,self.base_hash)
            model.to(self.device).eval()
            self.models[capability]=model
        return self.models[capability]

    def selected_model(self, history, user):
        capability=route_capability(history,user)
        return capability,self.model_for(capability) if capability else None
