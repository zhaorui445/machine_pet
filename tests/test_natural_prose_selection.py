import json
from pathlib import Path
import tempfile
import unittest

from petlm import natural_prose_selection as selection


class NaturalProseSelectionTests(unittest.TestCase):
    def test_topic_keywords_do_not_approve_institutional_or_injected_text(self):
        texts = ['召开学习工作会议，首先阅读，然后理解。',
                 '万博平台提款需要实名认证。首先做饭，然后回家。',
                 '植物风水改变运势，因为空气，所以温度。']
        for text in texts:
            with self.subTest(text=text):
                genre, reason, _ = selection.classify(text)
                self.assertIsNone(genre)
                self.assertIsNotNone(reason)

    def test_natural_process_is_only_a_candidate(self):
        genre, reason, _ = selection.classify('我先洗菜，接着切菜，然后回家做饭。')
        self.assertEqual(genre, 'daily')
        self.assertIsNone(reason)

    def test_new_sources_preserve_text_split_and_pending_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            text = '我先洗菜，接着切菜，然后回家做饭。'
            row = {'id': 'a'*64, 'text': text, 'split': 'validation', 'origin': 'https://unknown.example/story',
                   'tokens_with_eos': 24, 'review_status': 'not_human_reviewed'}
            for split in ('train', 'validation', 'test'):
                for group in ('source-restricted', 'other-sources-hold'):
                    content = json.dumps(row, ensure_ascii=False)+'\n' if split=='validation' and group=='other-sources-hold' else ''
                    (source/f'{group}-{split}.jsonl').write_text(content, encoding='utf-8')
            selection.save(source/'manifest.json', {'kind': 'new_fineweb_prose_supplement_audit_v1',
                'files': {p.name: selection.checksum(p) for p in source.iterdir()}})
            output = root/'result'
            selection.build([source], output)
            review = json.loads((output/'review.json').read_text(encoding='utf-8'))['rows']
            self.assertEqual(review[0]['text'], text)
            self.assertEqual(review[0]['split'], 'validation')
            self.assertEqual(review[0]['source_tier'], 'source_review_hold')
            self.assertFalse(review[0]['training_eligible'])
            self.assertIsNone(review[0]['human_decision'])
            self.assertEqual((output/'train-candidates.jsonl').read_text(), '')
            (source/'other-sources-hold-validation.jsonl').write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                selection.build([source], root/'invalid')
            self.assertFalse((root/'invalid').exists())


if __name__ == '__main__':
    unittest.main()
