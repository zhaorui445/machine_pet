"""Minimal LoRA adapters for PetLM linear layers."""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        if not isinstance(base,nn.Linear):
            raise TypeError('LoRALinear requires nn.Linear.')
        if rank<1 or alpha<=0 or not 0<=dropout<1:
            raise ValueError('Invalid LoRA rank, alpha or dropout.')
        self.base=base
        self.rank=rank
        self.alpha=float(alpha)
        self.scaling=self.alpha/rank
        self.dropout=nn.Dropout(dropout)
        self.lora_a=nn.Linear(base.in_features,rank,bias=False)
        self.lora_b=nn.Linear(rank,base.out_features,bias=False)
        self.lora_a.to(device=base.weight.device,dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device,dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight,a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad=False

    def forward(self,x):
        return self.base(x)+self.lora_b(self.lora_a(self.dropout(x)))*self.scaling


def inject_lora(model, rank=8, alpha=16.0, dropout=0.05):
    """Replace every transformer linear layer and return their module names."""
    targets=[]
    for block_index,block in enumerate(model.blocks):
        replacements=(
            ('qkv',block,'qkv'),('proj',block,'proj'),
            ('mlp.0',block.mlp,'0'),('mlp.2',block.mlp,'2'),
        )
        for suffix,parent,attribute in replacements:
            original=parent[int(attribute)] if attribute.isdigit() else getattr(parent,attribute)
            adapted=LoRALinear(original,rank,alpha,dropout)
            if attribute.isdigit():
                parent[int(attribute)]=adapted
            else:
                setattr(parent,attribute,adapted)
            targets.append(f'blocks.{block_index}.{suffix}')
    for name,parameter in model.named_parameters():
        parameter.requires_grad=('.lora_a.' in name or '.lora_b.' in name)
    return targets


def adapter_state_dict(model):
    return {name:tensor.detach().cpu() for name,tensor in model.state_dict().items()
            if '.lora_a.' in name or '.lora_b.' in name}


def load_adapter_state(model,state):
    if not isinstance(state,dict):
        raise TypeError('Adapter state must be a dictionary.')
    current=model.state_dict()
    missing=[name for name in current if ('.lora_a.' in name or '.lora_b.' in name) and name not in state]
    unexpected=[name for name in state if name not in current]
    if missing or unexpected:
        raise RuntimeError(f'Adapter state mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}')
    current.update(state)
    model.load_state_dict(current)


def apply_adapter_checkpoint(model,checkpoint,base_checkpoint_sha256):
    """Validate and apply one standalone adapter checkpoint to its exact base model."""
    if checkpoint.get('kind')!='petlm_lora_adapter_v1':
        raise RuntimeError('Unsupported adapter checkpoint.')
    if checkpoint.get('base_checkpoint_sha256')!=base_checkpoint_sha256:
        raise RuntimeError('LoRA adapter does not match the base checkpoint.')
    if checkpoint.get('config')!=model.config:
        raise RuntimeError('LoRA adapter model configuration does not match the base model.')
    settings=checkpoint.get('lora',{})
    try:
        targets=inject_lora(model,settings['rank'],settings['alpha'],settings['dropout'])
    except KeyError as exc:
        raise RuntimeError(f'LoRA adapter settings are incomplete: {exc}') from exc
    if settings.get('targets')!=targets:
        raise RuntimeError('LoRA adapter target layers do not match the base model.')
    load_adapter_state(model,checkpoint.get('adapter_state'))
    return settings


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
