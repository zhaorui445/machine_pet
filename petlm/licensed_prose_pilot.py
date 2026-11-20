"""Prepare bounded source pilots from contiguous prose, with original chapter isolation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

import sentencepiece as spm

from .foundation_diagnostics import compact
from .pretrain_corpus import prohibited_character_ratio, repeated_shingle_ratio
from .pretrain_increment import canonical_hashes, PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL
from .prose_supplement import save
from .run import checksum
from .wikimedia_prose import normalize, fingerprints, deduplicate, END

STRUCTURE = re.compile(r'!\[|\$|:cite:|:numref:|:eqref:|:ref:|^\s*(?:[-+*]\s|\d+[.)]\s|>|\||:label:|:width:|:height:|===)|<!--|\{\{|https?://', re.M)
DEPENDENCY = re.compile(r'^(?:这[个些种时就里]|该|它|其|因此|所以|然而|但是|此外|上述|接下来|最后|以上|前面|在以上)|下[图表式]|上[图表式]|如图|以上步骤|以上编程语言|例题[一二三四五六七八九十]|上述代码|下面的代码|上面的代码|前一节|上一节|以下代码')
CONTENT = re.compile(r'自杀|自伤|轻生|治疗|药物|处方|服药|毒品|枪支|弹药|爆炸物|色情|博彩|赌博|贷款|保留所有权利|禁止转载|未经授权')


def render_prose(block):
    if STRUCTURE.search(block):
        return None
    # Allow ordinary inline emphasis and text links, never splice across figures or code.
    text = re.sub(r'\[([^\]]+)\]\([^\s)]+\)', r'\1', block)
    text = re.sub(r'\[\*\*(.*?)\*\*\]', r'\1', text)
    text = re.sub(r'</?(?:u|em|strong)>', '', text)
    if re.search(r'[<>]|`[^`]*[(){}=;][^`]*`', text):
        return None
    text = re.sub(r'`([^`]+)`', r'\1', text).replace('**','')
    text = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<=[\u3400-\u9fff，。；：、])\s*\n\s*(?=[\u3400-\u9fff])', '', text)
    text = normalize(re.sub(r'\s+', ' ', text))
    if not text or re.search(r'[\[\]{}|`\\]|\ufffd', text):
        return None
    if any(p.search(text) for p in (PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL)) or CONTENT.search(text):
        return None
    if not END.search(text) or prohibited_character_ratio(text):
        return None
    norm = compact(text)
    if len(re.findall(r'[\u3400-\u9fff]', norm))/max(1,len(norm)) < .70:
        return None
    return text


def contiguous_prose(markdown):
    """Keep adjacent original paragraphs together; any excluded block ends the run."""
    chunks, paragraph, section, fence = [], [], '', None

    def emit_block():
        if paragraph:
            chunks.append((section,'\n'.join(paragraph)))
            paragraph.clear()

    for line in markdown.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            emit_block()
            chunks.append((section,None))
            if fence is None:
                fence=marker[1][0]
            elif marker[1][0]==fence:
                fence=None
            continue
        if fence:
            continue
        if re.match(r'^#{1,6}\s', line):
            emit_block()
            chunks.append((section,None))
            section=line.lstrip('# ').strip()
        elif not line.strip():
            emit_block()
        else:
            paragraph.append(line)
    emit_block()
    run, indices, current_section = [], [], ''
    for i,(heading,block) in enumerate(chunks+[(section,None)]):
        text=render_prose(block) if block is not None else None
        if text is None or (run and (heading!=current_section or sum(map(len,run))+len(text)>4000)):
            if run:
                yield current_section, indices, '\n\n'.join(run)
            run,indices=[],[]
        if text is not None:
            current_section=heading
            run.append(text)
            indices.append(i)


def quality_reason(text):
    if not 100<=len(text)<=4000:
        return 'length'
    if len(re.findall('[。！？]',text))<2:
        return 'few_sentences'
    if DEPENDENCY.search(text):
        return 'missing_context'
    if repeated_shingle_ratio(text)>.10:
        return 'repetition'
    for a,b in [('“','”'),('「','」'),('(',')')]:
        if text.count(a)!=text.count(b):
            return 'unbalanced_punctuation'
    return None


def chapter(source,path):
    parts=Path(path).parts
    return parts[1] if source=='hello_algo' else parts[0] if source.startswith('d2l') else '/'.join(parts[:2])


def split_map(selection):
    """Fix chapter split before looking at quality/results; never move a row later."""
    result={}
    for source,paths in selection['files'].items():
        groups=sorted({chapter(source,p) for p in paths},key=lambda g:hashlib.sha256((source+':'+g).encode()).hexdigest())
        for i,group in enumerate(groups):
            # With only a few chapters, independence takes precedence over exact 90/5/5 ratios.
            split='validation' if len(groups)>=5 and i==0 else 'test' if len(groups)>=10 and i==1 else 'train'
            result[(source,group)]=split
    return result


def build(raw,base,tokenizer,selection_path,policy_path,output,verification_name='download-verification.json',limit=100):
    if output.exists():
        raise FileExistsError(output)
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    policies=json.loads(policy_path.read_text(encoding='utf-8'))
    verified=json.loads((raw/verification_name).read_text(encoding='utf-8'))
    if not verified['complete'] or verified['selection_sha256']!=checksum(selection_path):
        raise ValueError('Source selection changed')
    for source,policy in policies['sources'].items():
        for file,sha in policy['evidence_sha256'].items():
            if checksum(raw/source/file)!=sha:
                raise ValueError('License evidence changed')
        if policy['source_status']!='eligible_for_noncommercial_candidate_screening':
            raise ValueError('Source rights not verified for this pilot')
    parent=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    index=json.loads((base/'clean-index.meta.json').read_text(encoding='utf-8'))
    if index['snapshot_manifest_sha256']!=checksum(base/'manifest.json') or index['index_sha256']!=checksum(base/'clean-index.sqlite'):
        raise ValueError('Parent index changed')
    token_meta=json.loads((tokenizer/'manifest.json').read_text(encoding='utf-8'))
    if checksum(tokenizer/'tokenizer.model')!=token_meta['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    sp=spm.SentencePieceProcessor(model_proto=(tokenizer/'tokenizer.model').read_bytes())
    partitions=split_map(selection)
    pending,rejected,extracted=[],Counter(),Counter()
    db=sqlite3.connect((base/'clean-index.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        for receipt in verified['files']:
            source,path=receipt['source'],receipt['path']
            content=(raw/source/'text'/path).read_bytes()
            if hashlib.sha256(content).hexdigest()!=receipt['sha256']:
                raise ValueError('Raw document changed')
            policy=policies['sources'][source]
            group=chapter(source,path)
            for heading,indices,text in contiguous_prose(content.decode('utf-8')):
                extracted[source]+=1
                why=quality_reason(text)
                if why:
                    rejected[f'{source}/{why}']+=1
                    continue
                full,prefix=canonical_hashes(text)
                if db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1',(full,prefix)).fetchone():
                    rejected[f'{source}/parent_exact_prefix']+=1
                    continue
                ids=sp.encode(text)
                if sp.unk_id() in ids or compact(sp.decode(ids))!=compact(text):
                    rejected[f'{source}/tokenizer_roundtrip']+=1
                    continue
                origin=f'https://github.com/{policy["repository"]}/blob/{policy["commit"]}/{path}'
                pending.append({'id':full,'source':source,'text':text,'section':heading,'source_path':path,
                    'origin':origin,'history_url':origin.replace('/blob/','/commits/'),
                    'attribution':policy['attribution'],'license':policy['text_license'],'license_url':policy['license_url'],
                    'split':partitions[(source,group)],'split_group':source+':'+group,
                    'raw_document_sha256':receipt['sha256'],'original_block_indices':indices,
                    'tokens_with_eos':len(ids)+1,'human_decision':None,'reviewer':None,
                    'human_review_status':'pending','training_eligible':False})
    finally:
        db.close()
    # Bounded acquisition pilot: at most 100 candidates per source, preserving chapter split.
    bounded=[]
    for source in selection['files']:
        pool=sorted((r for r in pending if r['source']==source),key=lambda r:r['id'])
        # Keep held-out coverage then training; content hashes make selection deterministic.
        held=[r for r in pool if r['split']!='train'][:max(20,limit//5)]
        train=[r for r in pool if r['split']=='train'][:limit-len(held)]
        bounded.extend(held+train)
    wanted=set().union(*(fingerprints(r['text']) for r in bounded)) if bounded else set()
    overlap=set()
    for split in ('train','validation','test'):
        path=base/f'{split}.jsonl'
        if checksum(path)!=parent['files'][path.name]:
            raise ValueError('Parent text changed')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                overlap.update(fingerprints(json.loads(line)['text'])&wanted)
        print(f'Parent {split}: fingerprints checked',flush=True)
    # Exclude overlap with approved previous pilot and all its originating candidate corpora.
    history=Path('data/petlm/experiments/generation-repair-v1')
    additional_hashes=set()
    for name in ['licensed-natural-prose-r1','author-natural-prose-r1','licensed-source-expansion-r1']:
        directory=history/name
        if not directory.exists() or directory.resolve()==output.resolve():
            continue
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        for split in ('train','validation','test'):
            path=directory/f'{split}-candidates.jsonl'
            if checksum(path)!=manifest['files'][path.name]:
                raise ValueError('Prior candidate snapshot changed')
            for line in path.read_text(encoding='utf-8').splitlines():
                additional_hashes.update(fingerprints(json.loads(line)['text'])&wanted)
    kept,duplicates=deduplicate(bounded,overlap|additional_hashes)
    rejected.update(duplicates)
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__),output/'screening-source.py')
    shutil.copy2(policy_path,output/'source-policy.json')
    shutil.copy2(selection_path,output/'source-selection.json')
    counts,tokens,chapters=Counter(),Counter(),{}
    for split in ('train','validation','test'):
        with (output/f'{split}-candidates.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for row in kept:
                if row['split']!=split:
                    continue
                assert chapters.setdefault(row['split_group'],split)==split
                counts[f'{row["source"]}/{split}']+=1
                tokens[f'{row["source"]}/{split}']+=row['tokens_with_eos']
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    review=sorted([r for r in kept if r['split']!='test'],key=lambda r:(r['source'],r['split'],r['id']))
    save(output/'review.json',{'status':'pending_new_human_review','scope':'All train and validation candidates; test omitted','rows':review})
    lines=['# 新来源小批候选审核','',
        '这批不在此前20段审批范围内。正文保留原文连续段落，遇到代码/图片/公式/引用等边界断开，不跨缺失内容拼接。',
        '仅作预训练候选；不是自然多轮对话。train与validation按原章节隔离，test不展示。',
        '逐条检查语法、自然度、语义完整、事实、隐私、来源许可和第三方引用；填写通过/拒绝/待定及理由。','']
    for i,r in enumerate(review,1):
        lines += [f'## N{i:03d} · {r["source"]} · {r["split"]} · {r["section"]}',
            f'ID：{r["id"]}',f'来源：{r["origin"]}',f'署名：{r["attribution"]}',
            f'许可：{r["license"]} · {r["license_url"]}','',r['text'],'','人工结论：待审核','审核人：','理由：','']
    (output/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    chinese=sum(len(re.findall(r'[\u3400-\u9fff]',compact(r['text']))) for r in kept)
    chars=sum(len(compact(r['text'])) for r in kept)
    save(output/'manifest.json',{'kind':'licensed_chinese_source_pilot_v2','status':'pending_sample_inspection_and_human_review',
        'counts':dict(counts),'tokens_with_eos':dict(tokens),'extracted_contiguous_runs':dict(extracted),
        'before_bound_and_parent_sentence_check':dict(Counter(r['source'] for r in pending)),
        'rejected':dict(rejected),'review_rows':len(review),'chinese_character_ratio':chinese/max(1,chars),
        'split_groups':chapters,'source_download_verification_sha256':checksum(raw/verification_name),
        'parent_manifest_sha256':checksum(base/'manifest.json'),'tokenizer_sha256':token_meta['tokenizer_sha256'],
        'human_review_completed':False,'training_ready':False,'training_started':False,'physical_actuation':'disabled',
        'limits':['Character ratio is not a quality rating; new human review is still required.',
            'Small chapter-isolated validation is not an independent formal blind acceptance set.',
            'Fingerprint dedup is not semantic dedup or proof of novelty against every historical corpus.',
            'Natural explanatory textbook prose cannot replace everyday multi-turn conversations.'],
        'files':{p.name:checksum(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'counts':dict(counts),'tokens':dict(tokens),'review_rows':len(review)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verification-name',default='download-verification.json')
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--base',type=Path,default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    p.add_argument('--tokenizer',type=Path,default=Path('data/petlm/tokens-chinese-v26r1-strict-24k'))
    a=p.parse_args()
    build(a.raw,a.base,a.tokenizer,a.selection,a.policy,a.output,a.verification_name,a.limit)
