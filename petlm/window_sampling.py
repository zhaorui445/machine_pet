"""Deterministic packed-token windows with no repeated target positions."""
import numpy as np


def window_order(length, context, seed=None):
    if context < 1 or length <= context:
        raise ValueError('Insufficient tokens for a complete context plus next target.')
    offsets = np.arange(0, length - context, context, dtype=np.int64)
    if seed is not None:
        offsets = np.random.default_rng(seed).permutation(offsets)
    return offsets


def step_offsets(order, step, microstep, batch_size, accumulation):
    if min(step, microstep) < 0 or microstep >= accumulation or min(batch_size, accumulation) < 1:
        raise ValueError('Invalid step or batch coordinates.')
    start = (step * accumulation + microstep) * batch_size
    result = order[start:start + batch_size]
    if len(result) != batch_size:
        raise ValueError('Single-pass corpus exhausted; repeated sampling is prohibited.')
    return result


def validate_schedule(length, config, extending=False):
    mode = config.get('sampling', 'random_windows')
    if mode == 'random_windows':
        return None
    if mode != 'nonoverlapping_single_pass':
        raise ValueError('Unknown training sampling mode.')
    if extending:
        raise ValueError('Single-pass runs cannot extend an old schedule; use the fixed pilot.')
    order = window_order(length, config['context'], config['seed'])
    if config['steps'] * config['batch_size'] * config['accumulation'] > len(order):
        raise ValueError('Not enough unique token windows for the fixed single-pass schedule.')
    return order
