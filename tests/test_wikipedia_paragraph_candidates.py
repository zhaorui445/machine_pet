import json
from pathlib import Path
import tempfile
import unittest

from petlm.wikipedia_paragraph_candidates import verified_rows, paragraphs, extra_reason
from petlm.record_candidate_review import record
from petlm.wikimedia_prose import deduplicate


class WikipediaParagraphTests(unittest.TestCase):
    def test_citations_cannot_join_separate_paragraphs_or_erase_formula(self):
        raw = '第一段。<ref>引用\n\n引用结尾</ref>第二段。\n\n第三段<math>x</math>。\n\n完整段落。'
        valid = [r[2] for r in paragraphs(raw) if r[3] is None]
        self.assertEqual(valid, ['完整段落。'])

    def test_duplicate_page_and_truncated_batch_fail(self):
        row = {'id':'1', 'url':'source', 'eligible_index':0, 'raw_wikitext':'正文'}
        meta = {'url':'source', 'documents':2, 'start_offset':0, 'next_offset':2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'raw.jsonl'
            path.write_text(json.dumps(row)+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'batch count'):
                list(verified_rows(path, meta))
            path.write_text(json.dumps(row)+'\n'+json.dumps(dict(row, eligible_index=1))+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'page ID'):
                list(verified_rows(path, meta))

    def test_hidden_sentence_leakage_is_rejected(self):
        sentence = '植物的根会从土壤中吸收水分和养分，并通过茎部向上输送，在这个过程中叶片的蒸腾作用发挥了重要的作用'
        rows = [{'id':'a', 'split':'train', 'text':'学习植物。'+sentence+'。'},
                {'id':'b', 'split':'validation', 'text':'观察植物。'+sentence+'。'}]
        kept, stats = deduplicate(rows, set())
        self.assertEqual([r['id'] for r in kept], ['b'])

    def test_review_record_rejects_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest.json').write_text(json.dumps({'files':{'review.json':'bad'}}))
            (root/'review.json').write_text(json.dumps({'rows':[{'id':'1','text':'text'}]}))
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                record(root, 'pass', 'test', 'user statement')
            self.assertFalse((root/'user-review-record.json').exists())

    def test_context_and_long_quoted_material_are_held(self):
        self.assertIsNotNone(extra_reason('之后，他来到公园。'))
        self.assertIsNotNone(extra_reason('他说：“'+'引用文字'*15+'”。'))

    def test_subset_cannot_move_heldout_rows_or_change_text(self):
        from petlm.verify_paragraph_candidates import verify_subset_membership
        row={'id':'1','text':'原文','split':'validation','page_id':'10'}
        verify_subset_membership([row],[row])
        for changed in [dict(row,split='train'),dict(row,text='改写'),dict(row,id='2')]:
            with self.assertRaisesRegex(ValueError,'Subset'):
                verify_subset_membership([changed],[row])


if __name__ == '__main__':
    unittest.main()
