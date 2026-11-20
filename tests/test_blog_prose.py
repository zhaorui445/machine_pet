import unittest
from petlm.blog_prose import extract_html, paragraph_runs


class BlogProseTests(unittest.TestCase):
    def test_only_article_body(self):
        source='<nav><p>导航</p></nav><div class="post-body"><p>原文<strong>强调</strong>。</p></div><div class="comments"><p>评论及个人资料</p></div>'
        result=extract_html(source)
        self.assertEqual([r['original_text'] for r in result['paragraphs']],['原文强调。'])

    def test_quote_list_table_excluded(self):
        source='<div itemprop="articleBody"><p>甲。</p><blockquote><p>引文。</p></blockquote><ul><li><p>列表。</p></li></ul><table><tr><td><p>表格。</p></td></tr></table><p>乙。</p></div>'
        rows=extract_html(source)['paragraphs']
        self.assertEqual([r['original_text'] for r in rows],['甲。','乙。'])
        self.assertGreater(rows[1]['paragraph_index']-rows[0]['paragraph_index'],1)

    def test_image_and_code_do_not_become_prose(self):
        source='<div class="post-content"><p>图示<img src="x">残句。</p><p>前<code>code</code>后。</p><p>完整原文。</p></div>'
        rows=extract_html(source)['paragraphs']
        self.assertEqual([r['structural_rejection'] for r in rows],['non_prose_inline_or_image','non_prose_inline_or_image',None])

    def test_duplicate_responsive_body_not_counted_twice(self):
        one='<div class="post-body"><p>同一正文。</p></div>'
        result=extract_html(one+one)
        self.assertEqual(len(result['paragraphs']),1)
        self.assertEqual(result['identical_body_containers'],2)

    def test_differing_body_containers_fail_closed(self):
        with self.assertRaises(ValueError):
            extract_html('<div class="post-body"><p>甲。</p></div><div class="post-body"><p>乙。</p></div>')

    def test_heading_boundary_and_entities(self):
        rows=extract_html('<div class="post-body"><h2>标题</h2><p>甲 &amp; 乙。</p></div>')['paragraphs']
        self.assertEqual(rows[0]['section'],'标题')
        self.assertEqual(rows[0]['original_text'],'甲 & 乙。')

    def test_contiguous_original_short_paragraphs(self):
        doc=extract_html('<div class="post-body"><p>第一段。</p><p>第二段。</p><h2>另节</h2><p>第三段。</p></div>')
        runs=list(paragraph_runs(doc['paragraphs']))
        self.assertEqual([r['text'] for r in runs],['第一段。\n\n第二段。','第三段。'])

    def test_never_join_over_quote_or_private_detail(self):
        doc=extract_html('<div class="post-body"><p>甲。</p><blockquote><p>引文。</p></blockquote><p>乙。</p><p>联系 test@example.org。</p><p>丙。</p></div>')
        self.assertEqual([r['text'] for r in paragraph_runs(doc['paragraphs'])],['甲。','乙。','丙。'])

    def test_caption_credits_and_editorial_deletion(self):
        doc=extract_html('<div class="post-body"><p>甲。</p><div class="wp-caption"><img src="x"><p>图片署名。</p></div><p>乙。</p><p>删除<del>坏字</del>内容。</p><div class="gv-rss-footer"><p>作者信息。</p></div></div>')
        self.assertEqual([r['text'] for r in paragraph_runs(doc['paragraphs'])],['甲。','乙。'])


if __name__=='__main__':
    unittest.main()
