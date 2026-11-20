"""Isolation bridge from model output to a dry-run-only motion gateway."""
import json
import math


SCHEMA='petlm.motion_proposal.v1'
FIELDS={'schema','action','target','speed','confidence','requires_confirmation','reason'}
ACTIONS={'move','stop','estop','clarify','none'}


class MotionProposalBridge:
    """Validate suggestions; never expose or enable a physical actuator."""
    def __init__(self,gateway):
        if gateway.actuator_enabled is not False:
            raise RuntimeError('The proposal bridge accepts dry-run gateways only.')
        self.gateway=gateway

    @staticmethod
    def parse(model_output):
        if not isinstance(model_output,str) or len(model_output)>4096:
            raise ValueError('Model output must be a JSON string no longer than 4096 characters.')
        try:
            proposal=json.loads(model_output,parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f'Non-finite JSON number: {value}')))
        except json.JSONDecodeError as exc:
            raise ValueError('Model output must contain JSON only.') from exc
        if not isinstance(proposal,dict) or set(proposal)!=FIELDS:
            raise ValueError('Motion proposal schema mismatch.')
        if proposal['schema']!=SCHEMA or proposal['action'] not in ACTIONS:
            raise ValueError('Unknown proposal schema or action.')
        if not isinstance(proposal['reason'],str) or not proposal['reason'].strip() or len(proposal['reason'])>256:
            raise ValueError('Proposal reason must contain 1 to 256 characters.')
        confidence=proposal['confidence']
        if (not isinstance(confidence,(int,float)) or isinstance(confidence,bool)
                or not math.isfinite(confidence) or not 0<=confidence<=1):
            raise ValueError('Proposal confidence must be finite and between 0 and 1.')
        if not isinstance(proposal['requires_confirmation'],bool):
            raise ValueError('requires_confirmation must be boolean.')
        if proposal['action']=='move':
            target=proposal['target']
            speed=proposal['speed']
            if (not isinstance(target,list) or len(target)!=2
                    or not all(isinstance(value,(int,float)) and not isinstance(value,bool)
                               and math.isfinite(value) for value in target)):
                raise ValueError('Move proposal requires two finite target coordinates.')
            if (not isinstance(speed,(int,float)) or isinstance(speed,bool)
                    or not math.isfinite(speed) or speed<=0):
                raise ValueError('Move proposal speed must be positive and finite.')
            if proposal['requires_confirmation'] is not True:
                raise ValueError('Move proposals must require confirmation.')
        elif proposal['target'] is not None or proposal['speed'] is not None:
            raise ValueError('Only move proposals may contain target and speed.')
        return proposal

    def submit(self,model_output,command_id,now,policy_approved=False,
               operator_confirmed=False):
        proposal=self.parse(model_output)
        action=proposal['action']
        if action in ('clarify','none'):
            return {'status':'proposal_not_executed','action':action,'dry_run':True}
        if action=='move' and (policy_approved is not True or operator_confirmed is not True):
            return {'status':'proposal_blocked','action':'move','dry_run':True,
                    'reason':'policy_and_operator_confirmation_required'}
        kind=action
        fields={}
        if action=='move':
            fields={'target':proposal['target'],'speed':proposal['speed']}
        elif action in ('stop','estop'):
            fields={'reason':f'model_proposal:{proposal["reason"]}'}
        outcome=self.gateway.execute({'type':kind,'command_id':command_id,'time':now,**fields})
        return {'status':'submitted_to_dry_run_gateway','action':action,'dry_run':True,
                'actuator_enabled':False,'gateway_outcome':outcome}
