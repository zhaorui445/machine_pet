import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from petlm.build_daily_dialogue_v6r6r2 import build
from petlm.human_review_assignment import create_assignment
from petlm.record_batch_confirmation import record
from petlm.human_review_pipeline import read_jsonl
from petlm.train_daily_v6r6r2 import prepare


class DailyTrainingIsolationTests(unittest.TestCase):
    def setup_package(self, root):
        source, assigned, confirmed = (root / name for name in ('source', 'assigned', 'confirmed'))
        build(source)
        create_assignment(source, assigned, 'test-reviewer', 8)
        record(assigned, confirmed, 'test fixture: all qualified')
        tokenizer = root / 'tokenizer.model'
        tokenizer.write_bytes(b'test-tokenizer-hash-only')
        return source, confirmed, tokenizer

    def test_snapshot_only_reads_development_and_preserves_approval_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, confirmed, tokenizer = self.setup_package(root)
            opened = []

            def checked_read(path):
                self.assertNotIn('internal_test', Path(path).name)
                opened.append(str(path))
                return read_jsonl(path)

            with patch('petlm.train_daily_v6r6r2.read_jsonl', side_effect=checked_read):
                manifest = prepare(source, confirmed, root / 'snapshot', tokenizer)
            self.assertEqual(manifest['counts'], {'train': 24, 'validation': 8})
            self.assertFalse(set(manifest['train_ids']) & set(manifest['validation_ids']))
            self.assertFalse(manifest['internal_test_used'])
            self.assertFalse(manifest['independent_signed_item_review_complete'])
            self.assertEqual(manifest['approval_method'], 'user_package_level_statement')
            training = read_jsonl(root / 'snapshot/train.jsonl')
            validation = read_jsonl(root / 'snapshot/validation.jsonl')
            self.assertTrue(all(row['source_split'] == 'development' for row in training + validation))
            self.assertEqual(len({row['behavior_family'] for row in validation}), 8)

    def test_changed_candidate_or_confirmation_is_rejected_before_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, confirmed, tokenizer = self.setup_package(root)
            path = source / 'development-candidates.jsonl'
            path.write_text(path.read_text(encoding='utf8') + '\n', encoding='utf8')
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                prepare(source, confirmed, root / 'snapshot', tokenizer)
            self.assertFalse((root / 'snapshot').exists())


if __name__ == '__main__':
    unittest.main()
