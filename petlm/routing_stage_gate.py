"""Write a conservative stage gate from frozen evaluation evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .run import checksum


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_report(regression_dir, answer_review_manifest, formal_score_path,
                 simulation_path, dry_run_path):
    regression_path = regression_dir / 'report.json'
    regression = load_json(regression_path)
    review = load_json(answer_review_manifest)
    formal = load_json(formal_score_path)
    simulation = load_json(simulation_path)
    dry_run = load_json(dry_run_path)

    splits = regression['splits']
    dev = splits['development:application']
    internal = splits['internal_test:application']
    route_min = min(
        splits['development:model']['route_correct'] / splits['development:model']['count'],
        splits['internal_test:model']['route_correct'] / splits['internal_test:model']['count'],
    )
    minimum_hit = min(
        dev['minimum_elements_present'] / dev['count'],
        internal['minimum_elements_present'] / internal['count'],
    )

    review_complete = (
        review.get('status') == 'independent_human_review_complete'
        and review.get('usable_as_frozen_answer_key') is True
        and review.get('item_level_review') is True
        and review.get('independence_attested') is True
    )
    score_bound_to_review = (
        formal.get('review_manifest_sha256') == checksum(answer_review_manifest)
        and formal.get('answer_key_sha256') == review.get('reviewed_answer_key_sha256')
        and formal.get('isolation_verified') is True
    )
    generation_manifest_path = Path(formal['generation_manifest'])
    if checksum(generation_manifest_path) != formal.get('generation_manifest_sha256'):
        raise RuntimeError('Formal score does not match the frozen generation manifest hash.')
    generation = load_json(generation_manifest_path)
    benchmark_id = formal.get('benchmark_id') or generation.get('benchmark_id')
    formal_count = generation['count']
    formal_route_rate = generation['route_correct'] / formal_count
    model = formal['modes']['model']
    application = formal['modes']['application']
    formal_application_rate = application['passed'] / application['total']
    formal_model_rate = model['passed'] / model['total']

    simulation_passed = simulation.get('all_passed') is True
    dry_run_passed = (
        dry_run.get('all_passed') is True
        and dry_run.get('dry_run') is True
        and dry_run.get('actuator_enabled') is False
        and dry_run.get('persistent_audit_chain_valid') is True
    )
    protected_software_passed = all((
        route_min == 1.0,
        minimum_hit >= 0.95,
        review_complete,
        score_bound_to_review,
        formal_application_rate >= 0.95,
        simulation_passed,
        dry_run_passed,
    ))
    hardware_motion_ready = protected_software_passed and all((
        formal_route_rate == 1.0,
        formal_model_rate >= 0.95,
        False,  # Physical emergency-stop and sensor validation are not complete.
    ))

    reasons = [
        f'Lowest development/internal router accuracy: {route_min:.3%}.',
        f'Lowest development/internal application requirement rate: {minimum_hit:.3%}.',
        f'Frozen {benchmark_id} application pass rate: {formal_application_rate:.3%}.',
        f'Frozen {benchmark_id} pure-model pass rate: {formal_model_rate:.3%}.',
        f'Frozen {benchmark_id} router accuracy: {formal_route_rate:.3%} ({generation["route_correct"]}/{formal_count}).',
        f'{benchmark_id} answer key completed independent item-level human review.' if review_complete
        else f'{benchmark_id} answer key review evidence is incomplete.',
        'Software motion simulation passed.' if simulation_passed
        else 'Software motion simulation did not pass.',
        'Dry-run gateway passed with actuators disabled and a valid audit chain.' if dry_run_passed
        else 'Dry-run gateway evidence is incomplete or unsafe.',
        'No physical braking, motor driver, emergency-stop wiring, or real sensor validation has been completed.',
    ]
    return {
        'kind': 'petlm_stage_gate_v3',
        'benchmark_id': benchmark_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'hardware_motion_blocked' if not hardware_motion_ready else 'eligible_for_supervised_hardware_motion_review',
        'protected_software_candidate_passed': protected_software_passed,
        'decision': 'Keep motors, wheels, GPIO/serial actuation, and autonomous navigation disabled. Continue software-only and non-actuating integration tests.',
        'reasons': reasons,
        'criteria': {
            'development_internal_router_accuracy': 1.0,
            'development_internal_application_requirement_rate': 0.95,
            'frozen_benchmark_independent_answer_review_required': True,
            'frozen_benchmark_application_pass_rate': 0.95,
            'frozen_benchmark_router_accuracy_for_hardware': 1.0,
            'frozen_benchmark_pure_model_pass_rate_for_hardware': 0.95,
            'software_motion_simulation_all_passed': True,
            'dry_run_gateway_all_passed_with_actuators_disabled': True,
            'physical_emergency_stop_and_sensor_validation_required': True,
        },
        'metrics': {
            'development_internal_router_accuracy_minimum': route_min,
            'development_internal_application_requirement_rate_minimum': minimum_hit,
            'frozen_route_correct': generation['route_correct'],
            'frozen_route_total': formal_count,
            'frozen_route_accuracy': formal_route_rate,
            'frozen_model_passed': model['passed'],
            'frozen_model_total': model['total'],
            'frozen_model_pass_rate': formal_model_rate,
            'frozen_application_passed': application['passed'],
            'frozen_application_total': application['total'],
            'frozen_application_pass_rate': formal_application_rate,
            'motion_simulation_all_passed': simulation_passed,
            'dry_run_gateway_all_passed': dry_run_passed,
        },
        'evidence': {
            'regression_report': {'path': str(regression_path), 'sha256': checksum(regression_path)},
            'answer_review_manifest': {'path': str(answer_review_manifest), 'sha256': checksum(answer_review_manifest)},
            'formal_score': {'path': str(formal_score_path), 'sha256': checksum(formal_score_path)},
            'generation_manifest': {'path': str(generation_manifest_path), 'sha256': checksum(generation_manifest_path)},
            'motion_simulation': {'path': str(simulation_path), 'sha256': checksum(simulation_path)},
            'dry_run_gateway': {'path': str(dry_run_path), 'sha256': checksum(dry_run_path)},
        },
        'limitations': (
            'Required-element scoring is deterministic and strict, but it is not a new semantic human rating. '
            'A perfect protected-application score does not validate unconstrained model quality or physical safety.'
        ),
        'frozen_benchmark_policy': (
            f'Do not tune this candidate against {benchmark_id} or rerun {benchmark_id} as a replacement result. '
            'Future changes require a separate development cycle and a new frozen blind set.'
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regression', type=Path, required=True)
    parser.add_argument('--answer-review-manifest', type=Path, required=True)
    parser.add_argument('--formal-score', type=Path, required=True)
    parser.add_argument('--simulation', type=Path, required=True)
    parser.add_argument('--dry-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    report = build_report(
        args.regression,
        args.answer_review_manifest,
        args.formal_score,
        args.simulation,
        args.dry_run,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
