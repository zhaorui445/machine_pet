import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from petlm import prose_supplement as audit


class TextTokenizer:
    def encode(self, text):
        return list(map(ord, text))

    def decode(self, ids):
        return ''.join(map(chr, ids))

    def unk_id(self):
        return -1


class ProseSupplementTests(unittest.TestCase):
    def prepare(self, root, texts, parent_texts=()):
        base, tokens = root/'base', root/'tokens'
        base.mkdir()
        tokens.mkdir()
        counts = {'train': 0, 'validation': 0, 'test': len(parent_texts)}
        for split in counts:
            with (base/f'{split}.jsonl').open('w', encoding='utf-8') as out:
                for text in parent_texts if split == 'test' else ():
                    out.write(json.dumps({'text': text}, ensure_ascii=False)+'\n')
        audit.save(base/'manifest.json', {'splits': counts, 'files': {
            f'{s}.jsonl': audit.checksum(base/f'{s}.jsonl') for s in counts}})
        with sqlite3.connect(base/'clean-index.sqlite') as conn:
            conn.execute('CREATE TABLE hashes (full TEXT PRIMARY KEY, prefix TEXT)')
            for text in parent_texts:
                conn.execute('INSERT INTO hashes VALUES (?, ?)', audit.canonical_hashes(text))
        conn.close()
        audit.save(base/'clean-index.meta.json', {
            'snapshot_manifest_sha256': audit.checksum(base/'manifest.json'),
            'index_sha256': audit.checksum(base/'clean-index.sqlite')})
        (tokens/'tokenizer.model').write_bytes(b'fake-test-tokenizer')
        audit.save(tokens/'manifest.json', {'tokenizer_sha256': audit.checksum(tokens/'tokenizer.model')})
        raw = root/'raw.jsonl'
        with raw.open('w', encoding='utf-8') as out:
            for i, text in enumerate(texts):
                out.write(json.dumps({'id': str(i), 'text': text, 'row_index': 527000+i,
                    'url': 'https://www.cas.cn/example', 'license': 'odc-by',
                    'dataset': 'HuggingFaceFW/fineweb-2', 'config': 'cmn_Hani',
                    'split': 'train', 'language': 'cmn', 'language_script': 'Hani'}, ensure_ascii=False)+'\n')
        audit.save(raw.with_suffix('.metadata.json'), {
            'sha256': audit.checksum(raw), 'dataset': 'HuggingFaceFW/fineweb-2',
            'config': 'cmn_Hani', 'split': 'train', 'license': 'odc-by',
            'start_offset': 527000, 'next_offset': 527000+len(texts),
            'documents': len(texts), 'rows_scanned': len(texts), 'skipped': 0})
        return raw, base, tokens

    def build(self, raw, base, tokens, output, splits=None):
        # Isolate integrity and partitioning logic from linguistic heuristics.
        with patch.object(audit.spm, 'SentencePieceProcessor', return_value=TextTokenizer()), \
                patch.object(audit, 'clean_document', side_effect=lambda t, *a: (t, None, {})), \
                patch.object(audit, 'prose_reason', return_value=None), \
                patch.object(audit, 'split_for', side_effect=lambda key: (splits or {}).get(key, 'train')):
            return audit.build([raw], base, tokens, output)

    def test_rejects_tampered_raw_before_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, base, tokens = self.prepare(root, ['自然中文。'])
            with raw.open('a', encoding='utf-8') as out:
                out.write('\n')
            with self.assertRaises(ValueError):
                self.build(raw, base, tokens, root/'output')
            self.assertFalse((root/'output').exists())

    def test_parent_test_duplicate_and_long_paragraph_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = '长段落中的共同内容。'*20
            parent = '原有文档开头。'*50+'\n'+shared
            paragraph_overlap = '新增文档不同开头。'*50+'\n'+shared
            new = '全新日常叙述。'*50
            raw, base, tokens = self.prepare(root, [parent, paragraph_overlap, new], [parent])
            parent_hashes = {p.name: audit.checksum(p) for p in base.iterdir()}
            report = self.build(raw, base, tokens, root/'output')
            self.assertEqual(report['strict_rejections']['parent_or_batch_exact_prefix_duplicate'], 1)
            self.assertEqual(report['prose_rejections']['parent_exact_prefix_or_long_paragraph_overlap'], 1)
            self.assertEqual(report['candidate_counts'], {'source-restricted/train': 1})
            self.assertEqual(report['review_rows'], 1)
            self.assertFalse(report['training_started'])
            self.assertFalse(report['human_review_completed'])
            self.assertEqual(parent_hashes, {p.name: audit.checksum(p) for p in base.iterdir()})

    def test_validation_wins_new_cross_split_paragraph_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = '应当保留在验证集的共同段落。'*10
            train = '训练文档的独特开头。'*50+'\n'+shared
            validation = '验证文档的独特开头。'*50+'\n'+shared
            raw, base, tokens = self.prepare(root, [train, validation])
            splits = {audit.canonical_hashes(validation)[0]: 'validation'}
            report = self.build(raw, base, tokens, root/'output', splits)
            self.assertEqual(report['candidate_counts'], {'source-restricted/validation': 1})
            self.assertEqual(report['prose_rejections']['new_candidate_exact_prefix_or_long_paragraph_duplicate'], 1)
            review = json.loads((root/'output'/'review.json').read_text(encoding='utf-8'))
            self.assertIsNone(review['rows'][0]['decision'])
            self.assertEqual(review['rows'][0]['split'], 'validation')


if __name__ == '__main__':
    unittest.main()
