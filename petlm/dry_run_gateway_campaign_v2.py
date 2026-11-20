"""Run the expanded non-actuating gateway boundary and recovery campaign."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import threading

from .dry_run_motion_gateway import DryRunMotionGateway
from .motion_proposal import MotionProposalBridge,SCHEMA
from .run import checksum


def cmd(kind,identifier,time_value,**fields):
    return {'type':kind,'command_id':identifier,'time':time_value,**fields}


def proposed(action,target=None,speed=None,confirmation=True):
    return json.dumps({'schema':SCHEMA,'action':action,'target':target,'speed':speed,
                       'confidence':0.9,'requires_confirmation':confirmation,
                       'reason':'campaign'},ensure_ascii=False)


def check(name,condition,details=None):
    return {'check':name,'passed':bool(condition),'details':details}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-output',type=Path,required=True)
    args=parser.parse_args()
    for path in (args.output,args.audit_output):
        if path.exists():
            raise FileExistsError(f'{path} exists; choose a new immutable path.')
    observed=[]

    persistent=DryRunMotionGateway(audit_path=args.audit_output)
    persistent.execute(cmd('heartbeat','persist-h1',0.0))
    persistent.execute(cmd('move','persist-m1',0.1,target=[1.0,0.0],speed=0.1))
    persistent.execute(cmd('estop','persist-e1',0.2,reason='campaign'))
    recovered=DryRunMotionGateway.recover(args.audit_output)
    observed.append(check('persistent_restart_replay',
        recovered.verify_audit_chain() and recovered.simulator.state.estop_latched
        and recovered.snapshot()['audit_head']==persistent.snapshot()['audit_head']))
    duplicate_blocked=False
    try:
        recovered.execute(cmd('heartbeat','persist-h1',0.3))
    except RuntimeError:
        duplicate_blocked=True
    observed.append(check('command_id_replay_blocked_after_restart',duplicate_blocked))

    boundary=DryRunMotionGateway(watchdog_timeout=0.5,max_coordinate=10)
    boundary.execute(cmd('heartbeat','b-h1',0.0))
    boundary.execute(cmd('move','b-m1',0.1,target=[1,0],speed=0.1))
    boundary.execute(cmd('tick','b-t1',0.5,dt=0.01))
    observed.append(check('watchdog_exact_boundary_allowed',not boundary.simulator.state.estop_latched))
    timeout=False
    try:
        boundary.execute(cmd('tick','b-t2',0.500001,dt=0.01))
    except RuntimeError:
        timeout=True
    observed.append(check('watchdog_over_boundary_estop',timeout and boundary.simulator.state.estop_latched))

    invalid=DryRunMotionGateway(max_coordinate=10)
    invalid.execute(cmd('heartbeat','i-h1',2.0))
    rejected={}
    candidates={
        'out_of_order':cmd('tick','i-order',1.0,dt=0.1),
        'oversized_coordinate':cmd('move','i-huge',2.0,target=[11,0],speed=0.1),
        'nan_coordinate':cmd('move','i-nan',2.0,target=[float('nan'),0],speed=0.1),
    }
    for name,value in candidates.items():
        try:
            invalid.execute(value)
        except (ValueError,RuntimeError):
            rejected[name]=True
        else:
            rejected[name]=False
    observed.extend(check(name,rejected[name]) for name in candidates)

    flood=DryRunMotionGateway(max_commands_per_window=2,rate_window=1.0)
    flood.execute(cmd('heartbeat','f-h1',0.0))
    flood.execute(cmd('heartbeat','f-h2',0.1))
    limited=False
    try:
        flood.execute(cmd('heartbeat','f-h3',0.2))
    except RuntimeError:
        limited=True
    estop=flood.execute(cmd('estop','f-e1',-100.0,reason='priority'))
    repeated_estop=flood.execute(cmd('estop','f-e1',-99.0,reason='priority_repeat'))
    observed.append(check('command_flood_limited',limited))
    observed.append(check('estop_bypasses_time_and_rate_rejection',estop['latched']))
    observed.append(check('duplicate_estop_is_idempotent',
                          repeated_estop['latched'] and repeated_estop['duplicate_command_id']))

    concurrent=DryRunMotionGateway()
    concurrent.execute(cmd('heartbeat','c-h1',0.0))
    barrier=threading.Barrier(3)
    def execute(value):
        barrier.wait()
        try:
            concurrent.execute(value)
        except RuntimeError:
            pass
    threads=(threading.Thread(target=execute,args=(cmd('move','c-m1',0.1,target=[1,0],speed=0.1),)),
             threading.Thread(target=execute,args=(cmd('estop','c-e1',0.1,reason='concurrent'),)))
    for thread in threads: thread.start()
    barrier.wait()
    for thread in threads: thread.join()
    observed.append(check('concurrent_estop_finishes_latched',
        concurrent.simulator.state.estop_latched and concurrent.simulator.state.speed==0
        and concurrent.verify_audit_chain()))

    proposal_gateway=DryRunMotionGateway()
    proposal_gateway.execute(cmd('heartbeat','p-h1',0.0))
    bridge=MotionProposalBridge(proposal_gateway)
    blocked=bridge.submit(proposed('move',[1,0],0.1),'p-blocked',0.1,
                          policy_approved=True,operator_confirmed=False)
    accepted=bridge.submit(proposed('move',[1,0],0.1),'p-move',0.1,
                           policy_approved=True,operator_confirmed=True)
    observed.append(check('model_proposal_requires_operator_confirmation',
                          blocked['status']=='proposal_blocked'))
    observed.append(check('approved_proposal_stays_dry_run',
        accepted['dry_run'] and not accepted['actuator_enabled']
        and not proposal_gateway.actuator_enabled))

    report={'kind':'petlm_dry_run_gateway_campaign_v2',
            'created_at':datetime.now(timezone.utc).isoformat(),
            'scope':'Software-only expanded boundary, persistence, concurrency, and proposal-isolation tests.',
            'dry_run':True,'actuator_enabled':False,
            'total':len(observed),'passed':sum(item['passed'] for item in observed),
            'failed':sum(not item['passed'] for item in observed),'checks':observed,
            'persistent_audit':str(args.audit_output),
            'persistent_audit_sha256':checksum(args.audit_output),
            'persistent_audit_entries':len(recovered.audit),
            'persistent_audit_head':recovered.snapshot()['audit_head'],
            'persistent_audit_chain_valid':recovered.verify_audit_chain(),
            'limitations':('No real motor, GPIO, serial, network, sensor hardware, power cut, '
                           'braking distance, wheel slip, or physical emergency-stop circuit was tested.')}
    report['all_passed']=(report['failed']==0 and report['persistent_audit_chain_valid']
                          and not report['actuator_enabled'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(report,report_sha256=checksum(args.output))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not report['all_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
