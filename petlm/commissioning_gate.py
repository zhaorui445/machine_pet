"""Combine model, benchmark-review, and motion-simulation readiness evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .run import checksum


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage-gate',type=Path,required=True)
    parser.add_argument('--answer-review',type=Path,required=True)
    parser.add_argument('--simulation',type=Path,required=True)
    parser.add_argument('--dry-run-gateway',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    stage=load(args.stage_gate)
    review=load(args.answer_review)
    simulation=load(args.simulation)
    gateway=load(args.dry_run_gateway) if args.dry_run_gateway else None
    reference_review_complete=review.get('status')=='independent_human_review_complete'
    simulation_passed=simulation.get('all_passed') is True
    gateway_passed=(gateway is not None and gateway.get('all_passed') is True
                    and gateway.get('actuator_enabled') is False
                    and gateway.get('persistent_audit_chain_valid') is True)
    model_gate_passed=stage.get('status')!='real_hardware_motion_blocked'
    hardware_ready=(reference_review_complete and simulation_passed and gateway_passed
                    and model_gate_passed)
    next_requirements=[]
    if not reference_review_complete:
        next_requirements.append(
            'Independent human item-level approval or rejection of all 400 v3 reference answers.')
    if not model_gate_passed:
        next_requirements.append('Meet the model/application regression thresholds recorded in the stage gate.')
    next_requirements.extend([
        'Prepare a physical emergency-stop and power-cut safety plan before any motor power is enabled.',
        'Validate real sensors, command transport authentication, stopping distance, and wheel-slip handling.',
        'Run supervised, restrained, low-speed hardware commissioning with measured stopping distance.'
    ])
    report={
        'kind':'petlm_commissioning_gate_v1','created_at':datetime.now(timezone.utc).isoformat(),
        'status':'hardware_motion_blocked' if not hardware_ready else 'eligible_for_supervised_low_speed_commissioning',
        'simulation_status':'passed_non_actuating_simulation' if simulation_passed else 'simulation_failed',
        'dry_run_gateway_status':('passed_persistent_non_actuating_gateway'
                                  if gateway_passed else 'gateway_evidence_missing_or_failed'),
        'v3_answer_review_status':review.get('status'),'model_stage_status':stage.get('status'),
        'allowed_now':['software simulation','non-actuating command interface tests','log and watchdog tests'],
        'prohibited_now':['motor power enable','wheel actuation','autonomous navigation','unsupervised physical testing'],
        'evidence':{
            'stage_gate':str(args.stage_gate),'stage_gate_sha256':checksum(args.stage_gate),
            'answer_review':str(args.answer_review),'answer_review_sha256':checksum(args.answer_review),
            'simulation':str(args.simulation),'simulation_sha256':checksum(args.simulation),
            'dry_run_gateway':str(args.dry_run_gateway) if args.dry_run_gateway else None,
            'dry_run_gateway_sha256':checksum(args.dry_run_gateway) if args.dry_run_gateway else None},
        'next_requirements':next_requirements}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
