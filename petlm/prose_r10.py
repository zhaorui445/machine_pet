"""Source-specific article containers, reusing the immutable r9 prose parser."""
from html import escape
from .blog_prose import Document, Node, VOID, extract_html, paragraph_runs, quality_reason, plain


def serialize(node):
    if isinstance(node,str):return escape(node)
    attrs=''.join(' '+k+'="'+escape(v or '',quote=True)+'"' for k,v in node.attrs.items())
    start='<'+node.tag+attrs+'>'
    if node.tag in VOID:return start
    return start+''.join(serialize(c) for c in node.children)+'</'+node.tag+'>'


def article_prose(html, source):
    doc=Document(html)
    nodes=list(doc.root.walk())
    def match(n):
        classes=set((n.attrs.get('class') or '').split())
        return ((source=='kawa' and 'entry-content' in classes)
                or (source=='jingyi' and n.attrs.get('id')=='post-content')
                or (source=='andypan' and 'article-content' in classes)
                or (source=='imfang' and 'post-content' in classes)
                or (source=='yolo' and 'markdown' in classes))
    bodies=[n for n in nodes if match(n)]
    if len(bodies)!=1:raise ValueError('Expected one explicit article body for '+source)
    body=bodies[0]
    title=next((n.attrs.get('content','') for n in nodes if n.tag=='meta' and n.attrs.get('property')=='og:title'),'')
    if not title:title=next((plain(n) for n in nodes if n.tag=='h1'),'')
    dates=[n.attrs['content'] for n in nodes if n.tag=='meta' and n.attrs.get('property')=='article:published_time']
    html_body=''.join(serialize(c) for c in body.children)
    # Article-specific rights exceptions take precedence over site-wide defaults.
    if 'by-nc-nd' in html_body.lower() or '禁止转载' in plain(body):
        raise ValueError('Article-specific restrictive rights declaration')
    parts=extract_html('<div class="post-body">'+html_body+'</div>')['paragraphs']
    return {'title':title,'dates':dates,'runs':list(paragraph_runs(parts)),
            'body_plain':plain(body),'paragraph_count':len(parts)}
