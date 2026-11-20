"""Read-only checkpoint comparison on newly authored development probes.

No old evaluation prompts/keys are opened. Every model uses its hash-matched
tokenizer and raw generation; no router, protection policy or fallback reply
can disguise the generated text. Proxies are not semantic human scores.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .run import checksum
from .human_review_pipeline import write_jsonl


ROOT = Path(__file__).resolve().parents[1]
PROBES = [
    {'id': 'continuation_river', 'kind': 'continuation',
     'prompt': '傍晚，河边的风渐渐凉了。小林收起画本，沿着石子路往家走。他看见路旁的小猫，',
     'reference': '便停下脚步，轻轻蹲在它旁边。小猫抬头看了看他，又慢慢走进了草丛。'},
    {'id': 'continuation_cooking', 'kind': 'continuation',
     'prompt': '做饭前，我先把茄子和豆腐洗好，切成小块。锅里的水烧开以后，',
     'reference': '我把食材放进锅里，等它们熟透，再按口味调味。一顿简单的家常饭就准备好了。'},
    {'id': 'continuation_science', 'kind': 'continuation',
     'prompt': '冰块放在温暖的房间里会慢慢融化。这是因为冰从周围吸收热量，',
     'reference': '温度逐渐升高，并由固态变成液态。这个变化没有产生新的物质。'},
    {'id': 'continuation_plan', 'kind': 'continuation',
     'prompt': '为了整理好书桌，小陈决定先把物品分成两类：一类是经常使用的，另一类是暂时用不上的。接着，',
     'reference': '他把常用的文具放在手边，把其余物品收进抽屉，最后擦干净桌面。'},
    {'id': 'chat_meal', 'kind': 'dialogue',
     'user': '晚餐想配米饭，手边有茄子和豆腐，可以做哪道家常菜？',
     'history': [], 'reference': '可以做茄子烧豆腐，配米饭吃。把茄子和豆腐煮熟，按自己的口味简单调味就好。'},
    {'id': 'chat_reason', 'kind': 'dialogue',
     'user': '雨停之后，为什么树叶上还会有水滴？',
     'history': [], 'reference': '雨水会留在叶片表面，一部分靠表面张力附着，所以雨停后不会立刻全部流走。'},
    {'id': 'chat_followup', 'kind': 'dialogue',
     'user': '那本书周四才到货，计划要怎么改？',
     'history': [('我打算周三开始看新买的书。', '可以先读一小章，再记下最感兴趣的一点。')],
     'reference': '把开始阅读的时间改到周四或之后，周三可以先整理阅读笔记本，不必等一本还没到的书。'},
    {'id': 'chat_topic_switch', 'kind': 'dialogue',
     'user': '给我的盆栽起两个有意思的名字。',
     'history': [('你能确认那句话的原始出处吗？', '暂时无法确认，我不编造来源。')],
     'reference': '可以叫“叶小圆”和“慢慢长”，一个可爱，一个像在鼓励它长大。'},
    {'id': 'extra_greeting', 'kind': 'dialogue', 'history': [],
     'user': '今天终于把房间收拾好了，想和你分享一下。',
     'reference': '收拾完房间应该轻松了不少！你最满意哪个地方的变化？'},
    {'id': 'extra_no_stove', 'kind': 'dialogue', 'history': [],
     'user': '午餐只有面包和能直接吃的熟牛肉，不用炉子能怎么搭配？',
     'reference': '可以把熟牛肉切片夹进面包，做一个简单的牛肉三明治。'},
    {'id': 'extra_preference', 'kind': 'dialogue', 'history': [],
     'user': '我不爱吃茄子，想用豆腐和青椒配饭，推荐一道菜。',
     'reference': '可以做青椒炒豆腐，豆腐煎至表面定型，再和青椒一起炒熟，配米饭吃。'},
    {'id': 'extra_story', 'kind': 'dialogue', 'history': [],
     'user': '用三句话讲一个小狗找雨伞的故事。',
     'reference': '小狗出门时忘了带伞，躲进了路边的小亭子。它在长椅下发现了自己的蓝雨伞。原来昨天散步时，它把伞落在这里了。'},
    {'id': 'extra_extract', 'kind': 'dialogue', 'history': [],
     'user': '小周把红杯子放进柜子，把蓝杯子放在桌上。桌上是什么颜色的杯子？',
     'reference': '桌上是蓝色的杯子。'},
    {'id': 'extra_correct', 'kind': 'dialogue',
     'history': [('周末想去公园骑车。', '可以挑天气好的时候出门。')],
     'user': '改主意了，我要在家拼积木，帮我想个小目标。',
     'reference': '那就按新计划，先用积木搭一座小桥，再试试看它能不能稳稳站住。'},
    {'id': 'extra_explain', 'kind': 'dialogue', 'history': [],
     'user': '为什么用毛巾能把桌上的水擦走？',
     'reference': '毛巾的纤维之间有许多细小空隙，水会沿这些空隙进入毛巾，所以能把桌面擦干。'},
    {'id': 'extra_listen', 'kind': 'dialogue', 'history': [],
     'user': '我的纸飞机怎么都飞不远，有点泄气。',
     'reference': '折了好多次还飞不远，确实会泄气。可以先放松一下，再看看两边机翼是不是对称。'},
]


def probe_prompt(probe):
    if probe['kind'] == 'continuation':
        return probe['prompt']
    return ''.join(f'用户：{u}\n助手：{a}\n' for u, a in probe['history']) + f'用户：{probe["user"]}\n助手：'


def load_state(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def inventory(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tokenizer_map = {}
    for path in sorted((ROOT / 'data/petlm').glob('*/tokenizer.model')):
        tokenizer_map.setdefault(checksum(path), []).append(str(path.relative_to(ROOT)))
    rows = []
    model_root = ROOT / 'data/petlm/models'
    for directory in sorted(path for path in model_root.iterdir() if path.is_dir()):
        states = {}
        for filename in ('best.pt', 'latest.pt'):
            path = directory / filename
            if not path.exists():
                continue
            row = {'id': f'{directory.name}/{filename}', 'path': str(path.relative_to(ROOT)),
                   'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            try:
                state = load_state(path)
                adapter = state.get('kind') == 'petlm_lora_adapter_v1'
                row.update(kind='adapter' if adapter else 'full', config=state['config'],
                           step=state.get('step'), tokenizer_sha256=state.get('tokenizer_sha256'),
                           tokenizer_paths=tokenizer_map.get(state.get('tokenizer_sha256'), []),
                           parent_checkpoint=state.get('parent_checkpoint'), parent_step=state.get('parent_step'),
                           base_checkpoint=state.get('base_checkpoint'), capability=state.get('capability'),
                           parameter_count=(sum(t.numel() for t in state['model'].values()) if not adapter else None))
                if adapter:
                    row['eligible_for_screen'] = directory.name == 'dialogue-v6r6r2-daily-r16-v1' and filename == 'best.pt'
                    row['reason'] = 'recent daily adapter comparator' if row['eligible_for_screen'] else 'specialized adapter, not a separate language-model starting point'
                else:
                    row['eligible_for_screen'] = bool(row['tokenizer_paths'])
                    row['reason'] = 'full checkpoint' if row['tokenizer_paths'] else 'matching tokenizer not found'
                # Best and latest often contain exactly the same weights. Do
                # not assume that merely from the step number: verify tensors.
                if filename == 'latest.pt' and 'best.pt' in states:
                    best = states['best.pt']
                    key = 'adapter_state' if adapter else 'model'
                    import torch
                    if (state.get('step') == best.get('step') and state[key].keys() == best[key].keys() and
                            all(torch.equal(state[key][name], best[key][name]) for name in state[key])):
                        row['eligible_for_screen'] = False
                        row['reason'] = 'identical weights to best.pt'
                if filename == 'best.pt':
                    states[filename] = state
                del state
            except Exception as exc:
                row.update(eligible_for_screen=False, reason=f'{type(exc).__name__}: {exc}')
            rows.append(row)
        states.clear()
        gc.collect()
    write_jsonl(output / 'inventory.jsonl', rows)
    write_jsonl(output / 'development-probes.jsonl', PROBES)
    report = {
        'kind': 'self_trained_checkpoint_comparison_plan_v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint_files': len(rows),
        'eligible_checkpoints': sum(row['eligible_for_screen'] for row in rows),
        'screen_probe_ids': [row['id'] for row in PROBES[:8]],
        'detail_probe_ids': [row['id'] for row in PROBES],
        'decoding': {'temperature': 0.3, 'max_new_tokens': 64, 'seed': 20261109},
        'inventory_sha256': checksum(output / 'inventory.jsonl'),
        'probes_sha256': checksum(output / 'development-probes.jsonl'),
        'scope': 'New development diagnostics only; neither blind acceptance nor training.',
        'no_policy_templates': True, 'no_generated_text_filter_or_fallback': True,
        'old_blind_data_used': False,
        'shortlisting_policy': 'Use byte-normalized reference loss as a diagnostic, then inspect raw generations; no automated semantic pass claim.',
    }
    (output / 'plan.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return rows


def raw_generate(model, tokenizer, prompt, seed, max_new_tokens=64, temperature=0.3):
    import torch
    from .dialogue_chat import sample_token
    torch.manual_seed(seed)
    prefix = tokenizer.encode(prompt)
    prefix = prefix[-(model.config['context'] - max_new_tokens):]
    ids = list(prefix)
    produced = []
    forbidden = {tokenizer.unk_id(), tokenizer.bos_id()}
    forbidden.update(index for index in range(tokenizer.vocab_size())
                     if tokenizer.id_to_piece(index).startswith('<0x'))
    device = next(model.parameters()).device
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1]
            token = sample_token(logits, produced, temperature, forbidden_tokens=forbidden)
            if token == tokenizer.eos_id():
                break
            ids.append(token)
            produced.append(token)
            text = tokenizer.decode(produced)
            if '\n用户：' in text or '\n助手：' in text:
                break
            if len(re.findall('[。！？!?]', text)) >= 2 and re.search('[。！？!?][”）)]*$', text):
                break
    return tokenizer.decode(produced), len(produced)


def reference_loss(model, tokenizer, prompt, reference):
    import torch
    from torch.nn import functional as F
    # No EOS term: token conventions differ between tokenizer generations.
    prefix, answer = tokenizer.encode(prompt), tokenizer.encode(reference)
    if not prefix or not answer or len(prefix) + len(answer) > model.config['context']:
        raise RuntimeError('Probe does not fit checkpoint context.')
    ids = prefix + answer
    device = next(model.parameters()).device
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor(answer, dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)[0, len(prefix)-1:]
        loss = F.cross_entropy(logits.float(), labels, reduction='sum').item()
    return {'reference_nll': loss, 'reference_tokens': len(answer),
            'reference_utf8_bytes': len(reference.encode('utf-8')),
            'reference_characters': len(reference),
            'reference_unknown_tokens': sum(index == tokenizer.unk_id() for index in answer),
            'reference_bits_per_utf8_byte': loss / math.log(2) / len(reference.encode('utf-8'))}


def mechanical_observations(text):
    compact = re.sub(r'\s+', '', text)
    grams = [compact[index:index+4] for index in range(max(0, len(compact)-3))]
    return {
        'empty': not compact,
        'source_search_template': any(term in text for term in ('论文', '检索', '不能核实确切出处')),
        'replacement_character': '\ufffd' in text,
        'chinese_ratio': len(re.findall('[\u3400-\u9fff]', compact)) / max(1, len(compact)),
        'repeated_four_gram_excess': sum(max(0, count-1) for count in Counter(grams).values()) / max(1, len(grams)),
        'semantic_score': None,
    }


def compare(output, phase, selected=()):
    import sentencepiece as spm
    import torch
    from .lora import apply_adapter_checkpoint
    from .model import PetLM
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    if (checksum(output / 'inventory.jsonl') != plan['inventory_sha256'] or
            checksum(output / 'development-probes.jsonl') != plan['probes_sha256']):
        raise RuntimeError('Fixed inventory/probes changed.')
    rows = [json.loads(line) for line in (output / 'inventory.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [row for row in rows if row['eligible_for_screen'] and (not selected or row['id'] in selected)]
    if selected and set(selected) != {row['id'] for row in rows}:
        raise ValueError('Selected checkpoint IDs are not all eligible.')
    phase_dir = output / phase
    if phase_dir.exists():
        raise FileExistsError(phase_dir)
    phase_dir.mkdir()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this comparison.')
    probes = PROBES[:8] if phase == 'screen' else PROBES
    aggregates = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        checkpoint_path = ROOT / row['path']
        if (checkpoint_path.stat().st_size != row['bytes'] or
                checkpoint_path.stat().st_mtime_ns != row['mtime_ns']):
            raise RuntimeError(f'Checkpoint changed since inventory: {row["id"]}')
        result = {'id': row['id'], 'path': row['path'], 'step': row['step'],
                  'tokenizer_sha256': row['tokenizer_sha256'], 'config': row['config'],
                  'parameter_count': row['parameter_count'], 'kind': row['kind']}
        try:
            state = load_state(checkpoint_path)
            tokenizer = spm.SentencePieceProcessor(model_proto=(ROOT / row['tokenizer_paths'][0]).read_bytes())
            if row['kind'] == 'adapter':
                base_path = Path(row['base_checkpoint'])
                base = load_state(base_path)
                model = PetLM(base['config'])
                model.load_state_dict(base['model'])
                apply_adapter_checkpoint(model, state, checksum(base_path))
                result['parameter_count'] = sum(p.numel() for p in model.parameters())
                del base
            else:
                model = PetLM(state['config'])
                model.load_state_dict(state['model'])
            del state
            model.to(device='cuda', dtype=torch.bfloat16).eval()
            predictions = []
            if phase == 'detail':
                screen_path = output / 'screen' / (row['id'].replace('/', '__') + '.json')
                screen_result = json.loads(screen_path.read_text(encoding='utf-8'))
                if (screen_result.get('status') != 'generated' or
                        screen_result.get('id') != row['id'] or
                        screen_result.get('tokenizer_sha256') != row['tokenizer_sha256']):
                    raise RuntimeError('Detail phase requires a successful matching screen result.')
                predictions.extend(screen_result['outputs'])
            already_generated = {case['probe_id'] for case in predictions}
            for probe_index, probe in enumerate(probes):
                if probe['id'] in already_generated:
                    continue
                prompt = probe_prompt(probe)
                text, tokens = raw_generate(model, tokenizer, prompt, plan['decoding']['seed'] + probe_index)
                predictions.append({
                    'probe_id': probe['id'], 'kind': probe['kind'], 'prompt': prompt,
                    'reference': probe['reference'], 'raw_output': text, 'generated_tokens': tokens,
                    'observations': mechanical_observations(text),
                    **reference_loss(model, tokenizer, prompt, probe['reference']),
                })
            result['outputs'] = predictions
            for kind in ('continuation', 'dialogue'):
                cases = [case for case in predictions if case['kind'] == kind]
                nll = sum(case['reference_nll'] for case in cases)
                byte_count = sum(case['reference_utf8_bytes'] for case in cases)
                result[f'{kind}_reference_bits_per_utf8_byte'] = nll / math.log(2) / byte_count
            result['source_template_count'] = sum(case['observations']['source_search_template'] for case in predictions)
            result['mean_repeated_four_gram_excess'] = sum(case['observations']['repeated_four_gram_excess'] for case in predictions) / len(predictions)
            result['empty_count'] = sum(case['observations']['empty'] for case in predictions)
            result['reference_unknown_tokens'] = sum(case['reference_unknown_tokens'] for case in predictions)
            result['status'] = 'generated'
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            if 'model' in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()
        result['seconds'] = time.perf_counter() - started
        result_path = phase_dir / (row['id'].replace('/', '__') + '.json')
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        aggregates.append({key: value for key, value in result.items() if key != 'outputs'})
        print(json.dumps({'progress': f'{index+1}/{len(rows)}', **aggregates[-1]}, ensure_ascii=False), flush=True)
    write_jsonl(phase_dir / 'summary.jsonl', aggregates)
    report = {'kind': 'checkpoint_development_comparison_phase_v1', 'phase': phase,
              'compared': len(rows), 'generated': sum(row['status'] == 'generated' for row in aggregates),
              'probe_count': len(probes), 'training_performed': False,
              'old_blind_data_used': False, 'human_semantic_scoring_complete': False,
              'note': 'Byte-normalized reference likelihood and repetition are diagnostic proxies; inspect the raw text before choosing.'}
    (phase_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def finalize(output):
    """Validate saved runs and record a provisional, non-deployment recommendation."""
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    for filename, key in (('inventory.jsonl', 'inventory_sha256'),
                          ('development-probes.jsonl', 'probes_sha256')):
        if checksum(output / filename) != plan[key]:
            raise RuntimeError(f'Fixed comparison input changed: {filename}')
    inventory_rows = [json.loads(line) for line in
                      (output / 'inventory.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id = {row['id']: row for row in inventory_rows}
    for row in inventory_rows:
        stat = (ROOT / row['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['bytes'], row['mtime_ns']):
            raise RuntimeError(f'Checkpoint changed: {row["id"]}')
    eligible = {row['id'] for row in inventory_rows if row['eligible_for_screen']}
    result_hashes = {}
    for phase, expected_count, expected_probes in (('screen', 96, PROBES[:8]), ('detail', 13, PROBES)):
        paths = sorted((output / phase).glob('*pt.json'))
        if len(paths) != expected_count:
            raise RuntimeError(f'Unexpected result count: {phase}')
        found = set()
        for path in paths:
            result = json.loads(path.read_text(encoding='utf-8'))
            if (result['status'] != 'generated' or result['id'] not in eligible or
                    [case['probe_id'] for case in result['outputs']] != [p['id'] for p in expected_probes] or
                    result['tokenizer_sha256'] != by_id[result['id']]['tokenizer_sha256']):
                raise RuntimeError(f'Invalid comparison output: {path}')
            found.add(result['id'])
            result_hashes[str(path.relative_to(output))] = checksum(path)
        if phase == 'screen' and found != eligible:
            raise RuntimeError('Screen results do not cover eligible models.')

    def candidate(checkpoint_id, tokenizer_path):
        row = by_id[checkpoint_id]
        token_hash = checksum(ROOT / tokenizer_path)
        if token_hash != row['tokenizer_sha256']:
            raise RuntimeError('Candidate tokenizer mismatch.')
        return {'checkpoint': row['path'], 'checkpoint_sha256': checksum(ROOT / row['path']),
                'step': row['step'], 'parameter_count': row['parameter_count'],
                'tokenizer': tokenizer_path, 'tokenizer_sha256': token_hash}

    base = candidate('chinese-v8-97m-extended-24k/latest.pt',
                     'data/petlm/tokens-chinese-v8-50m-frozen/tokenizer.model')
    control = candidate('dialogue-chinese-v8-v14-natural80/best.pt',
                        'data/petlm/tokens-chinese-v8-50m-frozen/tokenizer.model')
    freeze = json.loads((ROOT / 'data/petlm/evaluation/v6r5-candidate-freeze.json').read_text(encoding='utf-8'))
    runtime_keys = ('base_checkpoint', 'tokenizer', 'safety_adapter', 'memory_adapter',
                    'motion_adapter', 'format_adapter', 'behavior_state_code', 'router_code',
                    'chat_code', 'evaluation_code')
    frozen_hashes = {}
    for key in runtime_keys:
        entry = freeze['files'][key]
        actual = checksum(ROOT / entry['path'])
        if actual != entry['sha256']:
            raise RuntimeError(f'Frozen runtime asset differs from manifest: {key}')
        frozen_hashes[key] = actual
    receipt = {
        'kind': 'self_trained_checkpoint_comparison_recommendation_v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'no_checkpoint_demonstrated_normal_conversation',
        'provisional_further_pretraining_candidate': base,
        'full_model_reference_likelihood_control': control,
        'selection_basis': 'Agent inspection of new development generations plus diagnostic likelihood; no human acceptance score.',
        'selection_confidence': 'low; no robust semantic winner demonstrated',
        'training_started': False, 'application_replaced': False,
        'old_blind_data_used': False, 'human_review_complete': False,
        'inventory_files_checked': len(inventory_rows), 'screen_completed': 96, 'detail_completed': 13,
        'new_generation_count': 96 * 8 + 13 * 8,
        'inventory_sha256': plan['inventory_sha256'], 'probes_sha256': plan['probes_sha256'],
        'result_sha256': result_hashes,
        'frozen_runtime_assets_match_manifest': True, 'frozen_runtime_asset_sha256': frozen_hashes,
        'limitations': ['16 new development probes only; not blind acceptance.',
                       'Single seed and temperature; BF16 inference.',
                       'Existing sampler uses top-k=30, repetition penalty=1.15, trigram blocking and token masks.',
                       'Generation capped at 64 tokens or two terminal sentences; cannot evaluate three-sentence compliance.',
                       'Reference likelihood is not semantic correctness.',
                       'No exhaustive overlap audit against historical training corpora was performed.',
                       'All models remain substantially incoherent; provisional candidate is for foundational training experiments only.'],
    }
    (output / 'recommendation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return {key: value for key, value in receipt.items() if key not in ('result_sha256', 'frozen_runtime_asset_sha256')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('inventory', 'screen', 'detail', 'finalize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', default=[])
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory(args.output)
    elif args.command == 'finalize':
        print(json.dumps(finalize(args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(compare(args.output, args.command, args.checkpoint), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
