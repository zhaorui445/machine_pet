import unittest

import torch
from torch import nn

from petlm.lora import (adapter_state_dict, apply_adapter_checkpoint, inject_lora,
                        load_adapter_state, trainable_parameter_count)
from petlm.model import PetLM


def toy_model():
    torch.manual_seed(7)
    return PetLM({'vocab_size':32,'width':16,'layers':2,'heads':4,'context':12})


class LoRATests(unittest.TestCase):
    def test_injection_trains_only_adapter_parameters(self):
        model=toy_model()
        targets=inject_lora(model,rank=2,alpha=4,dropout=0)
        trainable=[name for name,parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(len(targets),8)
        self.assertTrue(trainable)
        self.assertTrue(all('.lora_a.' in name or '.lora_b.' in name for name in trainable))
        self.assertEqual(trainable_parameter_count(model),
                         sum(parameter.numel() for parameter in model.parameters()
                             if parameter.requires_grad))

    def test_initial_adapter_preserves_base_output(self):
        model=toy_model().eval()
        ids=torch.tensor([[1,2,3,4]])
        with torch.no_grad():
            expected=model(ids)
        inject_lora(model,rank=2,alpha=4,dropout=0.2)
        model.eval()
        with torch.no_grad():
            actual=model(ids)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_adapter_state_round_trip(self):
        source=toy_model()
        inject_lora(source,rank=2,alpha=4,dropout=0)
        with torch.no_grad():
            for name,parameter in source.named_parameters():
                if '.lora_b.' in name:
                    parameter.fill_(0.125)
        state=adapter_state_dict(source)

        target=toy_model()
        inject_lora(target,rank=2,alpha=4,dropout=0)
        load_adapter_state(target,state)
        loaded=adapter_state_dict(target)
        self.assertEqual(state.keys(),loaded.keys())
        for name in state:
            torch.testing.assert_close(loaded[name],state[name])

    def test_rejects_incomplete_and_unexpected_state(self):
        model=toy_model()
        inject_lora(model,rank=2,alpha=4,dropout=0)
        state=adapter_state_dict(model)
        incomplete=dict(state)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaisesRegex(RuntimeError,'missing'):
            load_adapter_state(model,incomplete)
        unexpected=dict(state,unexpected=torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError,'unexpected'):
            load_adapter_state(model,unexpected)
        with self.assertRaisesRegex(TypeError,'dictionary'):
            load_adapter_state(model,[])

    def test_checkpoint_requires_exact_base_hash(self):
        source=toy_model()
        targets=inject_lora(source,rank=2,alpha=4,dropout=0)
        checkpoint={
            'kind':'petlm_lora_adapter_v1',
            'base_checkpoint_sha256':'correct-hash',
            'config':source.config,
            'lora':{'rank':2,'alpha':4,'dropout':0,'targets':targets},
            'adapter_state':adapter_state_dict(source),
        }
        with self.assertRaisesRegex(RuntimeError,'does not match'):
            apply_adapter_checkpoint(toy_model(),checkpoint,'wrong-hash')
        target=toy_model()
        settings=apply_adapter_checkpoint(target,checkpoint,'correct-hash')
        self.assertEqual(settings['rank'],2)
        self.assertEqual(adapter_state_dict(target).keys(),checkpoint['adapter_state'].keys())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is not available')
    def test_new_adapter_parameters_follow_base_device(self):
        layer=nn.Linear(8,8,bias=False,device='cuda')
        from petlm.lora import LoRALinear
        adapted=LoRALinear(layer,rank=2,alpha=4,dropout=0)
        self.assertEqual(adapted.lora_a.weight.device.type,'cuda')
        self.assertEqual(adapted.lora_b.weight.device.type,'cuda')


if __name__=='__main__':
    unittest.main()
