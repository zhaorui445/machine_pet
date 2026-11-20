"""Build a larger high-quality Chinese pretraining snapshot from local licensed sources."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from opencc import OpenCC

from .data import BASE, clean, split_for
from .run import checksum

CONVERTER=OpenCC('t2s')
CJK=re.compile(r'[\u3400-\u9fff]')
SPAM=re.compile(r'博彩|六合彩|时时彩|送彩金|皇冠现金网|成人视频|裸聊|代开发票|加微信|联系QQ',re.I)
FINEWEB_SPAM=re.compile(
    r'百家乐|老虎机|娱乐城|赌博|赌球|现金游戏|体育投注|足球投注|波胆|盘口变化'
    r'|皇冠游戏|全讯网|棋牌(?:游戏|平台)|明升国际|博天堂|12bet|esball'
    r'|彩票[^。！？\n]{0,24}(?:投注|开奖|代理|预测)'
    r'|(?:办|买|出售|代办)[^。！？\n]{0,20}(?:毕业证|学历|文凭)'
    r'|(?:毕业证|学历|文凭)[^。！？\n]{0,24}(?:学信网|永久查询|Q\.?Q|加微)'
    r'|学信网[^。！？\n]{0,30}(?:永久查询|Q\.?Q|王老师)',re.I)
FINEWEB_SPAM_ORIGIN=re.compile(
    r'(?:^|[./_-])(?:y?12bet|casino|bocai|duchang|baijiale|laohuji|boshengguoji'
    r'|huangguan|yulecheng)(?:[./_-]|$)',re.I)
FINEWEB_PROMOTION=re.compile(
    r'在线咨询|免费咨询|点击咨询|专家热线|咨询热线|官方微信|品牌医院'
    r'|(?:医院|培训机构|招生机构)[^。！？\n]{0,80}(?:Q\.?Q|扣扣|微信|联系电话|预约电话|热线|地址)'
    r'|(?:专家|客服)[^。！？\n]{0,40}(?:Q\.?Q|扣扣|微信|电话|热线)',re.I)
# Keep HTML-like removal line-local.  A malformed '<' on one line must not
# consume ordinary prose up to a later '>' on another line.
HTML=re.compile(r'<[^>\r\n]{1,300}>')
WIKI_MARKUP=re.compile(r'\[编辑\]|\[編輯\]|\{\{[^{}]*\}\}|\[\[[^\]]+\]\]|thumb\s*\|[^\n]{0,180}',re.I)
TAIL_HEADING=re.compile(r'^(参考资料|参考文献|注释|注脚|外部链接|相关条目|延伸阅读|参见)\s*$',re.I)
NAVIGATION=re.compile(r'上一篇|下一篇|点击下载|友情链接|相关推荐')
URL=re.compile(r'https?://|www\.|@')
SENTENCE_END=re.compile(r'[。！？!?]')
STRICT_WEB_ARTIFACT=re.compile(
    r'(?i)(?:\.(?:jpg|jpeg|png|gif)\b|IMG[_-]\d|border\s*=|font-family\s*:|'
    r'下载次数|点击下方|class\s*=|style\s*=|\b(?:div|span)\s*>)')
STRICT_PROMOTION=re.compile(
    r'(?i)(?:报名|讲座|诚邀|欢迎加入|联系电话|咨询电话|客服电话|加微信|加\s*qq|'
    r'扫码|优惠价|限时优惠|地址[:：]|热线[:：]|预约|购买请|联系我们)')
STRICT_FORUM=re.compile(r'(?i)(?:大家好|小弟|求助|请问大家|哪位大大|发帖|回帖|下载附件)')
STRICT_JUNK=re.compile(
    r'(?i)(?:传奇|百家乐|老虎机|赌场|博彩|赌博|赌球|娱乐城|皇冠代理|真人游戏|'
    r'色情|裸聊|毛片|裸体|六合彩|彩票|投注|开奖|澳门威尼斯人|代理网|游戏网站|'
    r'股票推荐|贷款广告|高收益|返利|刷单|小说连载)')
STRICT_WIKI_TAIL=re.compile(r'^(?:Category:|分类:|图片|参考来源|外部链接)')


def to_simplified(text):
    for _ in range(3):
        converted=CONVERTER.convert(text)
        if converted==text:
            return text
        text=converted
    return text


def prohibited_character_ratio(text):
    count=0
    for character in text:
        if character.isspace():
            continue
        code=ord(character)
        category=unicodedata.category(character)
        prohibited=(
            0x1100<=code<=0x11ff or 0x3130<=code<=0x318f or 0xac00<=code<=0xd7af
            or 0x3040<=code<=0x30ff or 0x31f0<=code<=0x31ff
            or 0x2580<=code<=0x259f
            or 0x1f000<=code<=0x1faff or 0xe000<=code<=0xf8ff
            or 0x0370<=code<=0x052f or character=='\ufffd' or category in ('Cc','Cs','Co')
        )
        count+=int(prohibited)
    return count/max(1,len(text))


def collapse_duplicate_lines(text):
    kept=[]
    previous=None
    repeats=0
    for line in text.splitlines():
        normalized=re.sub(r'\s+','',line)
        if normalized and normalized==previous:
            repeats+=1
            continue
        kept.append(line)
        previous=normalized or previous
    return '\n'.join(kept),repeats


def repeated_shingle_ratio(text, width=12):
    compact=re.sub(r'\s+','',text)
    if len(compact)<width*4:
        return 0.0
    counts=Counter(compact[i:i+width] for i in range(0,len(compact)-width+1,3))
    repeated=sum(value-1 for value in counts.values() if value>1)
    return repeated/max(1,sum(counts.values()))


def clean_document(raw, source, profile='standard'):
    if source=='wikisource':
        return None,'classical_source',{}
    if SPAM.search(raw):
        return None,'spam_pattern',{}
    if source=='fineweb2' and FINEWEB_SPAM.search(raw):
        return None,'fineweb_spam_pattern',{}
    if source=='fineweb2' and FINEWEB_PROMOTION.search(raw):
        return None,'fineweb_promotional_contact',{}
    if profile=='strict' and source=='fineweb2':
        if STRICT_WEB_ARTIFACT.search(raw):
            return None,'strict_web_artifact',{}
        if STRICT_PROMOTION.search(raw):
            return None,'strict_promotional_or_contact',{}
    text=raw
    # Wiki templates can contain nested templates; remove innermost markup
    # repeatedly until another pass makes no change.
    while True:
        cleaned_markup=WIKI_MARKUP.sub(' ',text)
        if cleaned_markup==text:
            break
        text=cleaned_markup
    text=HTML.sub(' ',text)
    text=to_simplified(clean(text))
    # Traditional-to-simplified conversion can reveal prohibited phrases that
    # were not visible to the raw-text patterns (for example, 代開發票).
    if SPAM.search(text):
        return None,'spam_pattern',{}
    if source=='fineweb2' and FINEWEB_SPAM.search(text):
        return None,'fineweb_spam_pattern',{}
    if source=='fineweb2' and FINEWEB_PROMOTION.search(text):
        return None,'fineweb_promotional_contact',{}
    lines=[]
    stripped_tail=False
    for line in text.splitlines():
        line=line.strip(' *#\t')
        if source=='wikipedia' and TAIL_HEADING.fullmatch(line):
            stripped_tail=True
            break
        if profile=='strict' and source=='wikipedia' and STRICT_WIKI_TAIL.match(line):
            stripped_tail=True
            break
        if line:
            lines.append(line)
    text='\n'.join(lines)
    if profile=='strict' and source=='fineweb2':
        visible_lines=[line for line in lines if line.strip()]
        short_lines=sum(len(re.sub(r'\s+','',line))<24 for line in visible_lines)
        if len(visible_lines)>=12 and short_lines/max(1,len(visible_lines))>0.58:
            return None,'strict_fragmented_layout',{}
        if len(STRICT_FORUM.findall(text))>=2:
            return None,'strict_forum_mashup',{}
        if STRICT_JUNK.search(text):
            return None,'strict_junk_or_spam_topic',{}
        compact=re.sub(r'\s+','',text)
        latin=len(re.findall(r'[A-Za-z]',compact))/max(1,len(compact))
        digits=len(re.findall(r'\d',compact))/max(1,len(compact))
        separators=compact.count('|')+compact.count('｜')+compact.count('@@')
        if latin>0.12:
            return None,'strict_excess_latin',{}
        if digits>0.12:
            return None,'strict_excess_digits',{}
        if separators>=6:
            return None,'strict_fragment_separators',{}
    # Character-run normalization can make formerly different lines equal, so
    # perform it before adjacent-line deduplication to keep cleaning idempotent.
    text=re.sub(r'(.)\1{7,}',lambda match:match.group(1)*2,text)
    text,duplicate_lines=collapse_duplicate_lines(text)
    minimum=220 if source=='wikipedia' else 320
    if not minimum<=len(text)<=80_000:
        return None,'length',{}
    if prohibited_character_ratio(text)>0:
        return None,'unexpected_script_or_symbols',{}
    chinese_ratio=len(CJK.findall(text))/len(text)
    required_ratio=0.48 if source=='wikipedia' else 0.55
    if chinese_ratio<required_ratio:
        return None,'low_chinese_ratio',{}
    if len(URL.findall(text))>2:
        return None,'many_links',{}
    if len(NAVIGATION.findall(text))>3:
        return None,'navigation_text',{}
    if len(text)>600 and len(SENTENCE_END.findall(text))<2:
        return None,'insufficient_sentences',{}
    repetition=repeated_shingle_ratio(text)
    if repetition>0.22:
        return None,'high_shingle_repetition',{}
    blocks=[text[i:i+80] for i in range(0,len(text)-79,80)]
    if blocks and len(set(blocks))/len(blocks)<0.6:
        return None,'repeated_blocks',{}
    return text,None,{'duplicate_lines_removed':duplicate_lines,'wiki_tail_removed':stripped_tail,
                      'chinese_ratio':chinese_ratio,'repeated_shingle_ratio':repetition}


def dialogue_rows(path):
    manifest_path=path.parent/'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Dialogue provenance manifest is required: {manifest_path}')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    merged=manifest.get('kind')=='human_authored_chinese_dialogue_merge'
    dataset=manifest.get('source')
    revision=manifest.get('revision')
    license_note=manifest.get('license')
    if not merged and not all(isinstance(value,str) and value for value in (dataset,revision,license_note)):
        raise ValueError(f'Incomplete dialogue provenance in {manifest_path}')
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if row.get('split','train')!='train':
                raise ValueError(f'Non-training dialogue row found in {path}: {row.get("id")}')
            row_dataset=row.get('source_dataset') if merged else dataset
            row_revision=row.get('source_revision') if merged else revision
            row_license=row.get('source_license') if merged else license_note
            if not all(isinstance(value,str) and value for value in
                       (row_dataset,row_revision,row_license)):
                raise ValueError(f'Incomplete row provenance in {path}: {row.get("id")}')
            source='openassistant_'+row_dataset.rsplit('/',1)[-1].lower()
            text=(row['prompt']+row['answer']).strip()
            identifier=hashlib.sha256((row_dataset+'\0'+row['id']).encode()).hexdigest()
            yield {'id':identifier,'source':source,'split':'train','text':text,
                   'origin':f'{row_dataset}@{row_revision} pinned local training split',
                   'license':row_license,'dataset':row_dataset,'revision':row_revision}


def build(output, dialogue_paths, profile='standard', source_allowlist=None):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable snapshot path.')
    accepted=Counter()
    rejected=Counter()
    split_counts=Counter()
    quality=Counter()
    seen_full=set()
    seen_prefix=set()
    rows=[]
    provenance=[]
    with closing(sqlite3.connect(BASE/'corpus.sqlite')) as connection:
        query='SELECT hash,source,text FROM documents'
        parameters=()
        if source_allowlist:
            placeholders=','.join('?' for _ in source_allowlist)
            query+=f' WHERE source IN ({placeholders})'
            parameters=tuple(source_allowlist)
        query+=' ORDER BY hash'
        for original_key,source,raw in connection.execute(query,parameters):
            origin_rows=list(connection.execute(
                'SELECT origin,license,record_id FROM provenance WHERE hash=?',(original_key,)))
            if source=='fineweb2' and any(FINEWEB_SPAM_ORIGIN.search(origin or '')
                                          for origin,_,_ in origin_rows):
                rejected[(source,'fineweb_spam_origin')]+=1
                continue
            text,reason,signals=clean_document(raw,source,profile)
            if reason:
                rejected[(source,reason)]+=1
                continue
            canonical=re.sub(r'\s+','',text)
            full_hash=hashlib.sha256(canonical.encode()).hexdigest()
            prefix_hash=hashlib.sha256(canonical[:300].encode()).hexdigest()
            if full_hash in seen_full or prefix_hash in seen_prefix:
                rejected[(source,'duplicate_after_cleaning')]+=1
                continue
            seen_full.add(full_hash)
            seen_prefix.add(prefix_hash)
            split=split_for(full_hash)
            rows.append({'id':full_hash,'source':source,'split':split,'text':text})
            accepted[source]+=1
            split_counts[split]+=1
            quality['duplicate_lines_removed']+=signals['duplicate_lines_removed']
            quality['wiki_tails_removed']+=int(signals['wiki_tail_removed'])
            for origin,license_note,record_id in origin_rows:
                provenance.append({'id':full_hash,'source':source,'origin':origin,
                                   'license':license_note,'record_id':record_id})
    if isinstance(dialogue_paths,(str,Path)):
        dialogue_paths=[Path(dialogue_paths)]
    dialogue_paths=list(dialogue_paths or [])
    dialogue_added=0
    dialogue_counts=Counter()
    dialogue_provenance={}
    for dialogue_path in dialogue_paths:
        for row in dialogue_rows(dialogue_path):
            canonical=re.sub(r'\s+','',row['text'])
            full_hash=hashlib.sha256(canonical.encode()).hexdigest()
            if full_hash in seen_full:
                continue
            seen_full.add(full_hash)
            rows.append({key:row[key] for key in ('id','source','split','text')})
            provenance.append({key:row[key] for key in ('id','source','origin','license')})
            accepted[row['source']]+=1
            split_counts['train']+=1
            dialogue_added+=1
            dialogue_counts[row['dataset']]+=1
            dialogue_provenance[row['dataset']]={'revision':row['revision'],'license':row['license'],
                                                  'path':str(dialogue_path)}
    output.mkdir(parents=True)
    for split in ('train','validation','test'):
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in rows:
                if row['split']==split:
                    handle.write(json.dumps({k:row[k] for k in ('id','source','text')},ensure_ascii=False)+'\n')
    with (output/'provenance.jsonl').open('w',encoding='utf-8') as handle:
        for item in provenance:
            handle.write(json.dumps(item,ensure_ascii=False)+'\n')
    report={'sources':dict(accepted),'splits':dict(split_counts),
        'rejected':{f'{source}/{reason}':count for (source,reason),count in sorted(rejected.items())},
        'quality_actions':dict(quality),'dialogue_training_rows_added':dialogue_added,
        'dialogue_training_rows_by_dataset':dict(dialogue_counts),
        'dialogue_provenance':dialogue_provenance,
        'policy':{'quality_profile':profile,'source_allowlist':source_allowlist or 'all',
            'traditional_to_simplified':'OpenCC t2s','wikisource_included':False,
            'min_chars':{'wikipedia':220,'fineweb2':320},
            'min_chinese_ratio':{'wikipedia':0.48,'fineweb2':0.55},
            'max_prohibited_character_ratio':0.0,'max_repeated_shingle_ratio':0.22,
            'dedup':'exact normalized text and identical first 300 non-space characters',
            'dialogue_source':'Only explicitly supplied OpenAssistant training splits; validation/test excluded'},
        'limitations':['Heuristic quality checks can retain errors and reject useful text.',
            'FineWeb2 rights and content suitability still require source-level review.',
            'The local corpus remains far smaller than modern language-model pretraining corpora.']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dialogue-train',type=Path,action='append',
        help='Repeat for each pinned dialogue training JSONL; validation/test must not be supplied')
    parser.add_argument('--quality-profile',choices=('standard','strict'),default='standard')
    parser.add_argument('--source',choices=('fineweb2','wikipedia'),action='append',
        help='Restrict corpus sources; repeat for multiple sources')
    args=parser.parse_args()
    dialogue_paths=args.dialogue_train or [BASE/'dialogue'/'prepared-oasst1-chinese-v5'/'train.jsonl']
    build(args.output,dialogue_paths,args.quality_profile,args.source)


if __name__=='__main__':
    main()
