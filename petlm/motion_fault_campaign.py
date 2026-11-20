"""Run a deterministic non-actuating motion and fault-injection campaign."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .motion_simulator import FAULTS, MotionSafetySimulator


def scenario(name, run):
    try:
        details=run()
        return {'name':name,'passed':True,'details':details}
    except Exception as exc:
        return {'name':name,'passed':False,'error':f'{type(exc).__name__}: {exc}'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    results=[]

    def normal_motion():
        sim=MotionSafetySimulator(max_speed=0.20)
        sim.command_move((0.5,0),0.15)
        for _ in range(40): sim.tick(0.1)
        assert sim.state.mode=='stopped' and sim.state.stop_reason=='target_reached'
        assert abs(sim.state.x-0.5)<1e-9 and sim.state.speed==0
        return sim.snapshot()
    results.append(scenario('normal_motion_to_target',normal_motion))

    def speed_limit():
        sim=MotionSafetySimulator(max_speed=0.20)
        sim.command_move((1,0),2.0)
        assert sim.state.speed==0.20
        return sim.snapshot()
    results.append(scenario('speed_command_clamped',speed_limit))

    def manual_estop():
        sim=MotionSafetySimulator()
        sim.command_move((1,0),0.15)
        cycle=sim.state.control_cycles
        sim.emergency_stop('operator_button')
        assert sim.state.speed==0 and sim.state.estop_latched
        blocked=False
        try: sim.command_move((1,0),0.1)
        except RuntimeError: blocked=True
        assert blocked
        return {'stop_latency_control_cycles':sim.state.control_cycles-cycle,**sim.snapshot()}
    results.append(scenario('manual_emergency_stop_latched',manual_estop))

    for fault in FAULTS:
        def injected(fault=fault):
            sim=MotionSafetySimulator()
            sim.command_move((1,0),0.15)
            cycle=sim.state.control_cycles
            sim.inject_fault(fault)
            assert sim.state.speed==0 and sim.state.estop_latched
            assert sim.state.stop_reason==fault
            blocked=False
            try: sim.command_move((1,0),0.1)
            except RuntimeError: blocked=True
            assert blocked
            return {'fault':fault,'stop_latency_control_cycles':sim.state.control_cycles-cycle,
                    **sim.snapshot()}
        results.append(scenario(f'fault_injection_{fault}',injected))

    def recovery_interlock():
        sim=MotionSafetySimulator()
        sim.command_move((1,0),0.1)
        sim.inject_fault('sensor_offline')
        denied_without_operator=denied_while_faulted=False
        try: sim.reset_estop(False)
        except PermissionError: denied_without_operator=True
        try: sim.reset_estop(True)
        except RuntimeError: denied_while_faulted=True
        sim.restore_sensors()
        sim.reset_estop(True)
        assert denied_without_operator and denied_while_faulted
        assert not sim.state.estop_latched and sim.state.mode=='stopped' and sim.state.speed==0
        return {'denied_without_operator':denied_without_operator,
                'denied_while_faulted':denied_while_faulted,**sim.snapshot()}
    results.append(scenario('authorized_recovery_interlock',recovery_interlock))

    passed=sum(item['passed'] for item in results)
    report={'kind':'petlm_non_actuating_motion_fault_campaign_v1',
            'created_at':datetime.now(timezone.utc).isoformat(),
            'environment':'pure software simulation; no motor, GPIO, serial, network, or robot hardware access',
            'total':len(results),'passed':passed,'failed':len(results)-passed,
            'all_passed':passed==len(results),'scenarios':results,
            'scope':{'normal_motion':True,'speed_limit':True,'manual_emergency_stop':True,
                     'faults':list(FAULTS),'latched_reset_interlock':True},
            'limitations':('Simulation success does not validate physical braking distance, motor driver behavior, '
                           'power loss, sensor accuracy, timing jitter, wheel slip, or real emergency-stop wiring.'),
            'hardware_gate':'Remain blocked. Hardware commissioning requires an independent physical safety plan.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='scenarios'},ensure_ascii=False,indent=2))
    if not report['all_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
