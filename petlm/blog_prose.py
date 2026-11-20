"""Extract bounded original blog prose, without comments or reconstructed gaps."""
from html.parser import HTMLParser
import re

VOID = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
EXCLUDED = {'script','style','nav','footer','aside','form','blockquote','ul','ol','table','pre','figure','details','noscript'}
INLINE_UNSAFE = {'code','img','iframe','svg','math','sup','audio','video','input','button','del','s'}
EXCLUDED_CLASSES = {'wp-caption','wp-caption-text','gv-rss-footer','text-credits-container','footnotes'}


class Node:
    def __init__(self, tag, attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self):
        if self.tag in ('script','style'):
            return ''
        return ''.join(c.text() if isinstance(c, Node) else c for c in self.children)


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root=Node('root')
        self.stack=[self.root]
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        # HTML permits an omitted paragraph end tag.
        if tag=='p' and self.stack[-1].tag=='p':
            self.stack.pop()
        node=Node(tag,attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag,attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:
                del self.stack[i:]
                break

    def handle_data(self, text):
        self.stack[-1].children.append(text)


def plain(node):
    return re.sub(r'\s+', ' ', node.text()).strip()


def extract_html(html):
    """Return individual original paragraphs, retaining exclusion boundaries."""
    doc=Document(html)
    nodes=list(doc.root.walk())
    bodies=[n for n in nodes if n.attrs.get('itemprop')=='articleBody'
            or {'post-body','post-content'} & set(n.attrs.get('class','').split())]
    if not bodies:
        raise ValueError('No explicit article body container')
    signatures={plain(n) for n in bodies}
    if len(signatures)!=1:
        raise ValueError('Ambiguous article-body containers')
    title=next((n.attrs.get('content','') for n in nodes if n.tag=='meta' and n.attrs.get('property')=='og:title'),'')
    if not title:
        title=next((plain(n) for n in nodes if n.tag=='title'),'')
    dates=[n.attrs['datetime'] for n in nodes if n.tag=='time' and n.attrs.get('datetime')]
    rows=[]
    section=''
    sequence=0

    def visit(node):
        nonlocal section,sequence
        if node.tag in EXCLUDED or EXCLUDED_CLASSES & set(node.attrs.get('class','').split()):
            sequence+=1
            return
        if node.tag in {'h1','h2','h3','h4','h5','h6'}:
            section=plain(node)
            sequence+=1
            return
        if node.tag=='p':
            sequence+=1
            descendants=list(node.walk())
            why=None
            if any(n.tag in INLINE_UNSAFE for n in descendants):
                why='non_prose_inline_or_image'
            if any('footnote' in n.attrs.get('class','') for n in descendants):
                why='footnote_context'
            rows.append({'paragraph_index':sequence,'section':section,
                         'original_text':plain(node),'structural_rejection':why})
            return
        for child in node.children:
            if isinstance(child,Node):
                visit(child)
    visit(bodies[0])
    return {'title':title,'dates':dates,'identical_body_containers':len(bodies),'paragraphs':rows}


def content_reason(text):
    """Check even short paragraphs before they enter a contiguous run."""
    from .pretrain_increment import PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL
    if re.search(r'https?://|www\.|[<>|{}\\]|\ufffd|的的|了了|。。',text):
        return 'markup_or_language_defect'
    if any(p.search(text) for p in (PHONE,EMAIL,CN_ID,CARD,PRIVATE_DETAIL)):
        return 'personal_or_contact_pattern'
    if re.search(r'转载自|摘自|原作者|禁止转载|保留所有权利|未经.{0,8}许可|版权归|原文链接|译者|翻译自',text):
        return 'third_party_or_rights_notice'
    if re.search(r'[“「][^”」]{45,}[”」]',text):
        return 'long_inline_quotation'
    if re.search(r'自杀|自残|自伤|轻生|药物|治疗|处方|服药|诊断|感染|癌症|毒品|赌博|博彩|股票|证券|期货|贷款|借贷|枪支|弹药|炸药|强奸|色情|性交|死刑|诈骗|歧视|傻逼|傻屌',text):
        return 'outside_pilot_content_scope'
    return None


def paragraph_runs(blocks, max_chars=1200):
    """Keep only adjacent original paragraphs within one original section."""
    from .wikimedia_prose import normalize
    run=[]; indices=[]; section=''; last=-2
    for block in blocks+[None]:
        text=normalize(block['original_text']) if block else ''
        bad=(not block or block['structural_rejection'] or content_reason(text)
             or not re.search(r'[。！？][”’」』）)\"]*$',text))
        boundary=bad or (run and (block['section']!=section
                    or block['paragraph_index']!=last+1
                    or sum(map(len,run))+len(text)+2*len(run)>max_chars))
        if boundary and run:
            yield {'section':section,'original_paragraph_indices':indices,'text':'\n\n'.join(run)}
            run=[];indices=[]
        if not bad:
            section=block['section'];last=block['paragraph_index']
            run.append(text);indices.append(last)


def quality_reason(text):
    # Domain policy is scoped to a low-risk base-language pilot, not a censor.
    from .foundation_diagnostics import compact
    from .pretrain_corpus import repeated_shingle_ratio
    why=content_reason(text)
    if why:
        return why
    if not 100<=len(text)<=1600:
        return 'length'
    if len(re.findall('[。！？]',text))<2 or not re.search(r'[。！？][”’」』）)\"]*$',text):
        return 'incomplete_paragraph'
    if re.search(r'^(?:因此|所以|而且|此外|然而|但是|不过|最后|例如|比如|也就是|这[个些种时就里]|它|其|该)|[上下]图|如图|见图|如下|上述|下表|上表|前文|后文|这篇文章|本文|本篇',text):
        return 'context_dependency'
    norm=compact(text)
    if len(re.findall('[\u3400-\u9fff]',norm))/max(1,len(norm))<.70:
        return 'low_chinese_ratio'
    if repeated_shingle_ratio(text)>.10:
        return 'repetition'
    for a,b in [('“','”'),('「','」'),('(',')')]:
        if text.count(a)!=text.count(b):
            return 'unbalanced_punctuation'
    return None
