"""GPU end-to-end test with explicitly labeled synthetic fixtures, never source data."""
import json
from pathlib import Path
import sys
import uuid
import torch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from petlm.data import BASE, ROOT
from petlm.run import tokenize, train, generate


def main():
    root = BASE/'synthetic-verification'/uuid.uuid4().hex[:8]
    prepared = root/'prepared'
    prepared.mkdir(parents=True,exist_ok=True)
    for split in ('train','validation','test'):
        with (prepared/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for i in range(50):
                text = f'这是程序测试用的人工构造文本，编号{i}，集合{split}。我们一起学习科学，探索自然，阅读故事。' * 8
                handle.write(json.dumps({'text':text},ensure_ascii=False)+'\n')
    (prepared/'manifest.json').write_text(json.dumps({'sources':{'SYNTHETIC_TEST_ONLY':150},
        'missing_sources':['wikipedia','fineweb2','wikisource','clue2020'],
        'warning':'Synthetic fixture used ONLY to test software, not a corpus or useful trained model.'}),encoding='utf-8')
    tokenize(SimpleNamespace(data=prepared,tokens=root/'tokens',vocab_size=512))
    config = json.loads((ROOT/'configs/pet-small.json').read_text(encoding='utf-8'))
    config.update(steps=4,accumulation=1,eval_every=2)
    path = root/'test-config.json'
    path.write_text(json.dumps(config),encoding='utf-8')
    # This test uses the proposed width/depth with the fixture's smaller fitted vocabulary.
    args = SimpleNamespace(tokens=root/'tokens',config=path,steps=None,
        output=root/'model',resume=None,max_seconds=0.01)
    train(args)
    args.resume = root/'model/latest.pt'
    before = torch.load(args.resume,map_location='cpu',weights_only=True)['step']
    assert before < config['steps'], 'Test must stop before completion to exercise resumed training'
    args.max_seconds = 60
    train(args)
    after = torch.load(args.resume,map_location='cpu',weights_only=True)['step']
    assert after == config['steps'] and after > before
    args.resume = None
    args.extend_from = root/'model/latest.pt'
    args.output = root/'extension'
    args.steps = 6
    args.learning_rate = 0.00003
    train(args)
    extended = torch.load(args.output/'latest.pt',map_location='cpu',weights_only=True)
    assert extended['step'] == 6 and extended['phase_start'] == 4
    assert (root/'model/latest.pt').exists()
    generate(SimpleNamespace(tokens=root/'tokens',checkpoint=args.output/'latest.pt',prompt='我们一起学习',new_tokens=8))
    args.extend_from = None
    args.adapt_from = root/'extension/latest.pt'
    args.output = root/'adaptation'
    args.steps = 2
    train(args)
    adapted = torch.load(args.output/'latest.pt',map_location='cpu',weights_only=True)
    assert adapted['step'] == 2 and adapted['parent_step'] == 6
    assert adapted['parent_checkpoint'] == str(args.adapt_from.resolve())
    print('PASS: tokenizer, GPU pretraining, checkpoint resume and generation with synthetic fixtures.')


if __name__ == '__main__':
    main()
