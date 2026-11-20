"""Verify immutable prepared and tokenized corpus snapshots."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

from .pretrain_corpus import clean_document
from .run import checksum


SPLITS=('train','validation','test')


def text_key(text):
    return hashlib.sha256(''.join(text.split()).encode()).hexdigest()


def verify_prepared(root):
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    profile=manifest.get('policy',{}).get('quality_profile','standard')
    # Strict rejection rules are intentionally applied to raw source records
    # during snapshot construction. Reapplying them to already-cleaned rows
    # would make verification non-idempotent for contact/layout phrases.
    recheck_profile='standard' if profile=='strict' else profile
    ids={split:set() for split in SPLITS}
    texts={split:set() for split in SPLITS}
    counts=Counter()
    sources=Counter()
    quality_failures=[]
    file_hashes={}
    for split in SPLITS:
        path=root/f'{split}.jsonl'
        file_hashes[split]=checksum(path)
        with path.open(encoding='utf-8') as handle:
            for line_number,line in enumerate(handle,1):
                row=json.loads(line)
                if not all(isinstance(row.get(key),str) and row[key]
                           for key in ('id','source','text')):
                    raise ValueError(f'{path}:{line_number}: invalid row schema')
                if row['id'] in ids[split]:
                    raise ValueError(f'{path}:{line_number}: duplicate id')
                ids[split].add(row['id'])
                texts[split].add(text_key(row['text']))
                counts[split]+=1
                sources[row['source']]+=1
                if row['source'] in ('wikipedia','fineweb2'):
                    cleaned,reason,_=clean_document(row['text'],row['source'],recheck_profile)
                    if reason or cleaned!=row['text']:
                        quality_failures.append({'split':split,'line':line_number,
                                                 'id':row['id'],
                                                 'reason':reason or 'not_idempotent'})
    overlaps={}
    for left,right in (('train','validation'),('train','test'),('validation','test')):
        overlaps[f'id_{left}_{right}']=len(ids[left]&ids[right])
        overlaps[f'text_{left}_{right}']=len(texts[left]&texts[right])
    if any(overlaps.values()):
        raise RuntimeError(f'Cross-split overlap found: {overlaps}')
    if quality_failures:
        raise RuntimeError(f'Prepared rows fail current quality policy: {quality_failures[:5]}')
    expected=manifest.get('splits') or manifest.get('counts')
    if expected and any(expected.get(split)!=counts[split] for split in SPLITS):
        raise RuntimeError(f'Manifest count mismatch: expected {expected}, observed {dict(counts)}')
    return {'root':str(root),'manifest_sha256':checksum(manifest_path),
        'file_sha256':file_hashes,'counts':dict(counts),'sources':dict(sources),
        'cross_split_overlap':overlaps,'quality_recheck_failures':0,
        'status':'prepared_snapshot_verified'}


def verify_tokens(root,prepared):
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    tokenizer_path=root/'tokenizer.model'
    if checksum(tokenizer_path)!=manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer hash mismatch.')
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    if tokenizer.vocab_size()!=manifest['vocab_size']:
        raise RuntimeError('Tokenizer vocabulary size mismatch.')
    report={'root':str(root),'manifest_sha256':checksum(manifest_path),
        'tokenizer_sha256':checksum(tokenizer_path),'vocab_size':tokenizer.vocab_size(),
        'splits':{}}
    for split in SPLITS:
        source=prepared/f'{split}.jsonl'
        binary=root/f'{split}.bin'
        if checksum(source)!=manifest['input_sha256'][split]:
            raise RuntimeError(f'{split} prepared input hash mismatch.')
        if checksum(binary)!=manifest['binary_sha256'][split]:
            raise RuntimeError(f'{split} binary hash mismatch.')
        if binary.stat().st_size%4:
            raise RuntimeError(f'{split} binary is not aligned uint32 data.')
        data=np.memmap(binary,mode='r',dtype='<u4')
        if len(data)!=manifest['tokens'][split]:
            raise RuntimeError(f'{split} token count mismatch.')
        maximum=int(data.max()) if len(data) else -1
        if maximum>=tokenizer.vocab_size():
            raise RuntimeError(f'{split} contains out-of-vocabulary token id {maximum}.')
        report['splits'][split]={'tokens':len(data),'bytes':binary.stat().st_size,
            'maximum_token_id':maximum,'sha256':checksum(binary)}
    report['status']='token_snapshot_verified'
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--tokens',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result={'prepared':verify_prepared(args.prepared)}
    if args.tokens:
        result['tokens']=verify_tokens(args.tokens,args.prepared)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
