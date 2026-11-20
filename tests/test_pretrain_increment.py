import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from petlm.pretrain_increment import extend
from petlm.pretrain_corpus import clean_document
from petlm.run import checksum


def document(seed):
    topics = (
        '科学观察需要记录条件、过程与结果，并说明结论适用的范围。',
        '学习新概念可以先比较实例，再用自己的话总结其中规律。',
        '地图利用比例尺、方向和图例表示现实世界的空间信息。',
        '植物根系固定植株并吸收水分，叶片利用光能合成养分。',
        '阅读历史材料时需要核对作者、年代、目的和保存过程。',
        '程序测试能够发现错误，修正后还应重新运行相同的用例。',
        '团队讨论应明确问题，区分事实、推测以及个人价值判断。',
        '声音来自物体振动，介质性质会影响声音传播的速度。',
        '均衡饮食需要考虑食物种类、数量以及个人活动水平。',
        '实验操作必须遵守安全规范，异常情况应立即停止并记录。',
    )
    return seed + ''.join(
        f'{index}{text}这一段用于说明第{index}个主题，需要结合上下文判断并保留完整证据。'
        for index, text in enumerate(topics))


class PretrainIncrementTests(unittest.TestCase):
    def write_batch(self, root, name, rows, start):
        path = root / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                        encoding='utf-8')
        metadata = {'dataset': 'HuggingFaceFW/fineweb-2', 'config': 'cmn_Hani',
                    'start_offset': start, 'next_offset': start + len(rows),
                    'documents': len(rows), 'license': 'odc-by',
                    'endpoint': 'https://datasets-server.huggingface.co',
                    'sha256': checksum(path)}
        path.with_suffix('.metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
        return path

    def test_incremental_privacy_filter_dedup_and_index_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'base'
            base.mkdir()
            existing_raw = document('父快照内容。')
            existing, reason, _ = clean_document(existing_raw, 'fineweb2', 'strict')
            self.assertIsNone(reason)
            base_row = {'id': 'base-id', 'source': 'fineweb2', 'text': existing}
            (base / 'train.jsonl').write_text(json.dumps(base_row, ensure_ascii=False) + '\n',
                                              encoding='utf-8')
            for split in ('validation', 'test'):
                (base / f'{split}.jsonl').write_text('', encoding='utf-8')
            (base / 'provenance.jsonl').write_text(
                json.dumps({'id': 'base-id', 'source': 'fineweb2',
                            'origin': 'https://example.test/base', 'license': 'odc-by',
                            'record_id': 'base'}, ensure_ascii=False) + '\n', encoding='utf-8')
            manifest = {'sources': {'fineweb2': 1},
                        'splits': {'train': 1, 'validation': 0, 'test': 0},
                        'rejected': {}, 'quality_actions': {}, 'policy': {}}
            (base / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

            accepted = document('新增合格内容。')
            raw = self.write_batch(root, 'batch-one', [
                {'id': 'duplicate', 'text': existing_raw, 'url': 'https://example.test/duplicate',
                 'license': 'odc-by'},
                {'id': 'private', 'text': document('联系电话是13800138000。'),
                 'url': 'https://example.test/private', 'license': 'odc-by'},
                {'id': 'accepted', 'text': accepted, 'url': 'https://example.test/accepted',
                 'license': 'odc-by'},
            ], 100)
            output = root / 'output-one'
            extend(base, raw, output)
            result = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(sum(result['splits'].values()), 2)
            self.assertEqual(result['latest_increment']['rejected']
                             ['duplicate_against_snapshot_or_batch'], 1)
            self.assertEqual(result['latest_increment']['rejected']
                             ['raw_personal_or_contact_information'], 1)
            self.assertTrue((output / 'clean-index.sqlite').is_file())
            self.assertEqual(len((output / 'increment-review-sample.jsonl')
                                 .read_text(encoding='utf-8').splitlines()), 1)

            raw_two = self.write_batch(root, 'batch-two', [
                {'id': 'accepted-again', 'text': accepted,
                 'url': 'https://example.test/accepted-again', 'license': 'odc-by'},
            ], 103)
            output_two = root / 'output-two'
            extend(output, raw_two, output_two)
            result_two = json.loads((output_two / 'manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(result_two['latest_increment']['index_bootstrapped'])
            self.assertEqual(sum(result_two['splits'].values()), 2)
            self.assertEqual(result_two['latest_increment']['rejected']
                             ['duplicate_against_snapshot_or_batch'], 1)

    def test_incremental_rejects_non_idempotent_cleaned_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'base'
            base.mkdir()
            existing, reason, _ = clean_document(document('父快照内容。'), 'fineweb2', 'strict')
            self.assertIsNone(reason)
            (base / 'train.jsonl').write_text(json.dumps(
                {'id': 'base-id', 'source': 'fineweb2', 'text': existing},
                ensure_ascii=False) + '\n', encoding='utf-8')
            for split in ('validation', 'test'):
                (base / f'{split}.jsonl').write_text('', encoding='utf-8')
            (base / 'provenance.jsonl').write_text('', encoding='utf-8')
            (base / 'manifest.json').write_text(json.dumps({
                'sources': {'fineweb2': 1},
                'splits': {'train': 1, 'validation': 0, 'test': 0},
                'rejected': {}, 'quality_actions': {}, 'policy': {},
            }), encoding='utf-8')
            raw = self.write_batch(root, 'batch', [
                {'id': 'non-idempotent', 'text': document('新增内容。'),
                 'url': 'https://example.test/page', 'license': 'odc-by'},
            ], 200)

            def clean_with_late_markup(text, source, profile='standard'):
                if profile == 'strict':
                    return text + '<late-markup>', None, {}
                return text.replace('<late-markup>', ''), None, {}

            output = root / 'output'
            with patch('petlm.pretrain_increment.clean_document', clean_with_late_markup):
                extend(base, raw, output)
            result = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(result['latest_increment']['rejected']
                             ['quality_recheck_not_idempotent'], 1)
            self.assertEqual(sum(result['latest_increment']['accepted'].values()), 0)


if __name__ == '__main__':
    unittest.main()
