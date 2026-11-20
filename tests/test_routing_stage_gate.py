import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from petlm.routing_stage_gate import build_report
from petlm.run import checksum


class RoutingStageGateTests(unittest.TestCase):
    def _write_evidence(self, root, generation_hash=None):
        regression = root / 'regression'
        regression.mkdir()
        (regression / 'report.json').write_text(json.dumps({'splits': {
            'development:model': {'count': 10, 'route_correct': 10},
            'internal_test:model': {'count': 10, 'route_correct': 10},
            'development:application': {'count': 10, 'minimum_elements_present': 10},
            'internal_test:application': {'count': 10, 'minimum_elements_present': 10},
        }}), encoding='utf-8')
        review_path = root / 'review.json'
        review_path.write_text(json.dumps({
            'status': 'independent_human_review_complete',
            'usable_as_frozen_answer_key': True,
            'item_level_review': True,
            'independence_attested': True,
            'reviewed_answer_key_sha256': 'answer-hash',
        }), encoding='utf-8')
        generation_path = root / 'generation.json'
        generation_path.write_text(json.dumps({'count': 10, 'route_correct': 9}), encoding='utf-8')
        formal_path = root / 'formal.json'
        formal_path.write_text(json.dumps({
            'review_manifest_sha256': checksum(review_path),
            'answer_key_sha256': 'answer-hash',
            'isolation_verified': True,
            'generation_manifest': str(generation_path),
            'generation_manifest_sha256': generation_hash or checksum(generation_path),
            'modes': {
                'model': {'passed': 0, 'total': 10},
                'application': {'passed': 10, 'total': 10},
            },
        }), encoding='utf-8')
        simulation_path = root / 'simulation.json'
        simulation_path.write_text(json.dumps({'all_passed': True}), encoding='utf-8')
        dry_run_path = root / 'dry-run.json'
        dry_run_path.write_text(json.dumps({
            'all_passed': True,
            'dry_run': True,
            'actuator_enabled': False,
            'persistent_audit_chain_valid': True,
        }), encoding='utf-8')
        return regression, review_path, formal_path, simulation_path, dry_run_path

    def test_report_binds_review_and_frozen_score_but_blocks_hardware(self):
        with TemporaryDirectory() as temp:
            evidence = self._write_evidence(Path(temp))
            report = build_report(*evidence)
            self.assertTrue(report['protected_software_candidate_passed'])
            self.assertEqual(report['status'], 'hardware_motion_blocked')
            self.assertEqual(report['metrics']['frozen_route_accuracy'], 0.9)
            self.assertEqual(report['metrics']['frozen_application_pass_rate'], 1.0)

    def test_tampered_generation_manifest_is_rejected(self):
        with TemporaryDirectory() as temp:
            evidence = self._write_evidence(Path(temp), generation_hash='wrong')
            with self.assertRaisesRegex(RuntimeError, 'generation manifest hash'):
                build_report(*evidence)


if __name__ == '__main__':
    unittest.main()
