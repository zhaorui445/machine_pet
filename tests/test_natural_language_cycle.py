import json
from pathlib import Path
import tempfile
import unittest

from petlm.natural_language_cycle import Groups, anchors, cycle_progress, key, prose_rejection, prose_segments, prose_topic


class NaturalLanguageCycleTests(unittest.TestCase):
    def test_completed_training_is_not_reset_by_later_runtime_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pretrain').mkdir()
            (root / 'pretrain/report.json').write_text(json.dumps({
                'status': 'schedule_complete', 'completed_steps': 222,
            }), encoding='utf-8')
            result = cycle_progress(root, 'runtime_blocked')
            self.assertEqual(result['status'], 'pretraining_schedule_complete')
            self.assertTrue(result['training_started'])
            self.assertTrue(result['training_completed'])

    def test_preflight_and_baseline_are_not_completed_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(cycle_progress(root, 'ready')['status'], 'prepared_ready_for_tokenization')
            (root / 'token-verification.json').write_text(json.dumps({
                'status': 'token_snapshot_verified',
            }), encoding='utf-8')
            self.assertEqual(cycle_progress(root, 'ready')['status'], 'tokenized_ready_for_pretraining')
            (root / 'pretrain').mkdir()
            (root / 'pretrain/best.pt').touch()
            result = cycle_progress(root, 'ready')
            self.assertFalse(result['training_started'])
            self.assertFalse(result['training_completed'])
            self.assertEqual(result['status'], 'pretraining_initializing_or_interrupted')

    def test_shared_long_sentence_survives_punctuation_changes(self):
        sentence = '研究人员需要仔细记录每一次实验的环境条件和测量方法，再将不同时间取得的数据进行比较，判断观察到的变化是否能够支持原先提出的解释，并保留仍然不能确定的问题。'
        self.assertTrue(anchors(sentence + '\n第一个标题') & anchors(sentence.replace('，', ',') + '\n另一个标题'))

    def test_duplicate_groups_are_transitive(self):
        groups = Groups()
        groups.union('a', 'b')
        groups.union('c', 'b')
        self.assertEqual(groups.find('a'), groups.find('c'))

    def test_segments_discard_navigation_and_never_cut_sentences(self):
        sentence = '在书桌旁准备一本空白笔记，可以用来记录阅读过程中遇到的问题。'
        segments = list(prose_segments(sentence * 12 + '\n点击下载资料\n' + sentence * 12))
        self.assertTrue(segments)
        self.assertTrue(all(s.endswith('。') for s in segments))
        self.assertFalse(any('点击下载' in s for s in segments))

    def test_table_and_loop_text_are_not_natural_prose(self):
        self.assertIsNotNone(prose_rejection('1120年国王甲，1121年国王乙。' * 30))
        self.assertIsNotNone(prose_rejection('这是重复段落，没有更多内容。' * 40))
        self.assertEqual(key('中文，句子。'), key('中文,句子!'))

    def test_sampled_corpus_defects_are_rejected(self):
        prefix = '观察自然现象时需要认真记录不同的变化条件，并通过比较说明原因。' * 6
        self.assertEqual(prose_rejection(prefix + 'CopyRight©2016 网站声明。'), 'copyright_bibliography_or_markup')
        self.assertEqual(prose_rejection(prefix + '向量如, , 。'), 'missing_formula_or_text')
        self.assertEqual(prose_rejection(prefix + '提供整形美容注射服务。'), 'adult_spam_or_cosmetic_ad')
        self.assertEqual(prose_rejection(prefix + '海面温度冬季介于之间。'), 'missing_value_or_term')

    def test_mixture_targets_explanation_rather_than_name_lists(self):
        self.assertEqual(prose_topic('植物根系吸收水分，并从土壤获取营养。'), 'basic_science')
        self.assertIsNone(prose_topic('某某在王朝军队担任将领，后来负责学生教育。'))
        self.assertIsNone(prose_topic('甲国国王乙，乙国国王丙。'))


if __name__ == '__main__':
    unittest.main()
