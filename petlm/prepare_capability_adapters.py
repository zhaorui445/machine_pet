"""Split mixed dialogue SFT data into disjoint behavior-capability datasets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .dialogue_blind_v4 import read_jsonl, write_jsonl
from .run import checksum


CAPABILITIES={
    'safety_behavior':('risk_escalation','fact_correction','source_honesty'),
    'memory_management':('conflicting_history','memory_consent'),
    'motion_clarification':('route_clarification','tool_failure'),
    'format_control':('multi_constraint','step_confirmation','emotion_calibration'),
}
BEHAVIOR_GROUPS={'diverse_synthetic','confirmed_template','human_reviewed_behavior_v1',
                 'human_approved_behavior_v1','rule_authored_behavior_hardening_v1',
                 'rule_authored_capability_expansion_v2'}


def normalized(text):
    return re.sub(r'\s+','',text).casefold()


def public_prompt(row):
    history=''.join(f'用户：{turn["user"]}\n助手：{turn["assistant"]}\n'
                    for turn in row.get('history',[]))
    return history+f'用户：{row["user"]}\n助手：'


def split(source,output,frozen_prompts):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    source_manifest_path=source/'manifest.json'
    source_manifest=json.loads(source_manifest_path.read_text(encoding='utf-8'))
    frozen={normalized(public_prompt(row)) for row in read_jsonl(frozen_prompts)}
    source_rows={name:read_jsonl(source/f'{name}.jsonl')
                 for name in ('train','validation','test')}
    output.mkdir(parents=True)
    root_report={'kind':'petlm_capability_adapter_datasets_v1',
                 'source':str(source),'source_manifest_sha256':checksum(source_manifest_path),
                 'frozen_prompts':str(frozen_prompts),
                 'frozen_prompts_sha256':checksum(frozen_prompts),'capabilities':{}}
    for capability,families in CAPABILITIES.items():
        target=output/capability
        target.mkdir()
        counts={}
        family_counts=Counter()
        natural_count=0
        behavior_count=0
        all_prompts=set()
        for split_name,rows in source_rows.items():
            selected=[]
            for row in rows:
                family=row.get('behavior_family')
                is_behavior=(row.get('training_group') in BEHAVIOR_GROUPS or family is not None)
                if is_behavior and family not in families:
                    continue
                if is_behavior and family is None:
                    continue
                norm=normalized(row['prompt'])
                if norm in frozen:
                    raise RuntimeError(f'{capability} overlaps frozen v6 prompt: {row["id"]}')
                if norm in all_prompts:
                    raise RuntimeError(f'{capability} contains duplicate prompt: {row["id"]}')
                all_prompts.add(norm)
                copied=dict(row,adapter_capability=capability)
                selected.append(copied)
                if family:
                    family_counts[family]+=1
                    behavior_count+=1
                else:
                    natural_count+=1
            if not selected:
                raise RuntimeError(f'No {split_name} rows for {capability}.')
            write_jsonl(target/f'{split_name}.jsonl',selected)
            counts[split_name]=len(selected)
        absent=set(families)-set(family_counts)
        if absent:
            raise RuntimeError(f'{capability} lacks behavior families: {sorted(absent)}')
        report={'kind':'petlm_capability_adapter_dataset_v1','capability':capability,
                'behavior_families':list(families),'tokenizer_sha256':source_manifest['tokenizer_sha256'],
                'source':str(source),'source_manifest_sha256':checksum(source_manifest_path),
                'frozen_prompts_sha256':checksum(frozen_prompts),'frozen_prompt_overlap':0,
                'counts':counts,'natural_rows':natural_count,'behavior_rows':behavior_count,
                'counts_by_family':dict(sorted(family_counts.items())),
                'files':{name:checksum(target/f'{name}.jsonl')
                         for name in ('train','validation','test')}}
        (target/'manifest.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        root_report['capabilities'][capability]=report
    (output/'manifest.json').write_text(
        json.dumps(root_report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(root_report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-prompts',type=Path,required=True)
    args=parser.parse_args()
    split(args.source,args.output,args.frozen_prompts)


if __name__=='__main__':
    main()
