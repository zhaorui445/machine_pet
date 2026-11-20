import json
from pathlib import Path
import tempfile
import unittest

from petlm.prepare_capability_adapters import CAPABILITIES, split


class CapabilityAdapterDataTests(unittest.TestCase):
    def test_split_keeps_natural_replay_and_only_selected_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source'
            source.mkdir()
            (source/'manifest.json').write_text(
                json.dumps({'tokenizer_sha256':'tokenizer'}),encoding='utf-8')
            for split_name in ('train','validation','test'):
                rows=[{'id':f'{split_name}-natural','prompt':f'自然{split_name}',
                       'answer':'回答','training_group':'natural'},]
                for families in CAPABILITIES.values():
                    for family in families:
                        rows.append({'id':f'{split_name}-{family}','prompt':f'{split_name}-{family}',
                                     'answer':'回答','behavior_family':family,
                                     'training_group':'human_approved_behavior_v1'})
                (source/f'{split_name}.jsonl').write_text(
                    '\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n',encoding='utf-8')
            frozen=root/'prompts.jsonl'
            frozen.write_text(json.dumps({'id':'blind','history':[],'user':'未见问题'},
                                         ensure_ascii=False)+'\n',encoding='utf-8')
            output=root/'output'
            split(source,output,frozen)
            for capability,families in CAPABILITIES.items():
                rows=[json.loads(line) for line in
                      (output/capability/'train.jsonl').read_text(encoding='utf-8').splitlines()]
                observed={row.get('behavior_family') for row in rows if row.get('behavior_family')}
                self.assertEqual(observed,set(families))
                self.assertTrue(any(row['id'].endswith('-natural') for row in rows))


if __name__=='__main__':
    unittest.main()
