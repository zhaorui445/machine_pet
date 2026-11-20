"""Exercise the dry-run command gateway and persist its hash-chained audit log."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .dry_run_motion_gateway import DryRunMotionGateway
from .run import checksum


def cmd(kind,identifier,time_value,**fields):
    return {'type':kind,'command_id':identifier,'time':time_value,**fields}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    gateway=DryRunMotionGateway(watchdog_timeout=0.5,max_speed=0.15)
    observed=[]
    gateway.execute(cmd('heartbeat','cmd-001',0.0))
    gateway.execute(cmd('move','cmd-002',0.1,target=[1.0,0.0],speed=0.1))
    gateway.execute(cmd('tick','cmd-003',0.2,dt=0.1))
    observed.append({'check':'move_only_in_simulator','passed':not gateway.actuator_enabled and gateway.simulator.state.x>0})
    try:
        gateway.execute(cmd('tick','cmd-004',0.8,dt=0.1))
    except RuntimeError:
        observed.append({'check':'heartbeat_timeout_stops','passed':gateway.simulator.state.speed==0 and gateway.simulator.state.estop_latched and gateway.simulator.state.stop_reason=='watchdog_timeout'})
    else:
        observed.append({'check':'heartbeat_timeout_stops','passed':False})
    gateway.execute(cmd('estop','cmd-005',0.9,reason='operator_button'))
    blocked=False
    try:
        gateway.execute(cmd('move','cmd-006',0.95,target=[2.0,0.0],speed=0.1))
    except RuntimeError:
        blocked=True
    observed.append({'check':'latched_stop_blocks_motion','passed':blocked and gateway.simulator.state.speed==0})
    authorized_reset=False
    try:
        gateway.execute(cmd('reset','cmd-007',1.0,operator_id='operator-test',operator_authorized=False))
    except PermissionError:
        authorized_reset=True
    observed.append({'check':'unauthorized_reset_rejected','passed':authorized_reset and gateway.simulator.state.estop_latched})
    gateway.execute(cmd('fault','cmd-008',1.1,fault='sensor_offline'))
    observed.append({'check':'sensor_fault_latched','passed':gateway.simulator.state.estop_latched and gateway.simulator.state.speed==0})
    report={'kind':'petlm_dry_run_gateway_campaign_v1','created_at':datetime.now(timezone.utc).isoformat(),
            'dry_run':True,'actuator_enabled':gateway.actuator_enabled,
            'total':len(observed),'passed':sum(item['passed'] for item in observed),
            'failed':sum(not item['passed'] for item in observed),'checks':observed,
            'audit_chain_valid':gateway.verify_audit_chain(),'audit_entries':len(gateway.audit),
            'audit_head':gateway.snapshot()['audit_head'],'audit':gateway.audit,
            'limitations':'Software-only; no real motor, GPIO, serial, network, sensor hardware, or emergency-stop circuit accessed.'}
    report['all_passed']=report['passed']==report['total'] and report['audit_chain_valid'] and not report['actuator_enabled']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report['report_sha256']=checksum(args.output)
    print(json.dumps({k:v for k,v in report.items() if k!='audit'},ensure_ascii=False,indent=2))
    if not report['all_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
