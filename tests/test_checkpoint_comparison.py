import math
import unittest

import torch

from petlm.checkpoint_comparison import mechanical_observations, probe_prompt, reference_loss


class UniformModel(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab = vocab
        self.config = {'context': 32}

    def forward(self, ids):
        return torch.zeros((1, ids.shape[1], self.vocab), device=ids.device)


class ToyTokenizer:
    def __init__(self, merge):
        self.merge = merge

    def encode(self, text):
        return [1] if self.merge else [1] * len(text)

    def unk_id(self):
        return 0


class CheckpointComparisonTests(unittest.TestCase):
    def test_reference_metric_normalizes_bytes_instead_of_token_count(self):
        split = reference_loss(UniformModel(4), ToyTokenizer(False), '问', '甲乙')
        merged = reference_loss(UniformModel(16), ToyTokenizer(True), '问', '甲乙')
        self.assertEqual(split['reference_tokens'], 2)
        self.assertEqual(merged['reference_tokens'], 1)
        self.assertAlmostEqual(split['reference_bits_per_utf8_byte'],
                               merged['reference_bits_per_utf8_byte'], places=6)
        self.assertAlmostEqual(split['reference_bits_per_utf8_byte'], 4/6, places=6)

    def test_prompt_retains_followup_context_without_added_response_policy(self):
        probe = {'kind': 'dialogue', 'history': [('旧问题', '旧回答')], 'user': '新问题'}
        self.assertEqual(probe_prompt(probe), '用户：旧问题\n助手：旧回答\n用户：新问题\n助手：')
        self.assertEqual(probe_prompt({'kind': 'continuation', 'prompt': '清晨，'}), '清晨，')

    def test_mechanical_flags_never_claim_semantic_success(self):
        gibberish = mechanical_observations('菜或菜面和面食。')
        template = mechanical_observations('请用论文检索。')
        self.assertIsNone(gibberish['semantic_score'])
        self.assertFalse(gibberish['empty'])
        self.assertTrue(template['source_search_template'])


if __name__ == '__main__':
    unittest.main()
