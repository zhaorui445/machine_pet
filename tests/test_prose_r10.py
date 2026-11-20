import unittest
from petlm.prose_r10 import article_prose


class R10ExtractionTests(unittest.TestCase):
    def test_excludes_wordpress_comments(self):
        result=article_prose('<div class="entry-content"><p>正文甲。</p></div><article class="comment"><p>评论乙。</p></article>','kawa')
        self.assertEqual(result['runs'][0]['text'],'正文甲。')

    def test_preserves_boundaries_and_does_not_join_through_figure(self):
        result=article_prose('<div id="post-content"><p>第一段。</p><figure><p>图注。</p></figure><p>第二段。</p></div>','jingyi')
        self.assertEqual([r['text'] for r in result['runs']],['第一段。','第二段。'])

    def test_ambiguous_markdown_and_nd_license_rejected(self):
        with self.assertRaises(ValueError):article_prose('<div class="markdown"></div><div class="markdown"></div>','yolo')
        with self.assertRaises(ValueError):article_prose('<div class="entry-content"><p>正文。</p><a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">许可</a></div>','kawa')

    def test_html_entities_not_double_decoded(self):
        result=article_prose('<div class="article-content"><p>甲 &amp; 乙。</p></div>','andypan')
        self.assertEqual(result['runs'][0]['text'],'甲 & 乙。')

    def test_empty_class_attribute_does_not_break_body_detection(self):
        result=article_prose('<div class="markdown"><span class></span><p>正文。</p></div>','yolo')
        self.assertEqual(result['runs'][0]['text'],'正文。')


if __name__=='__main__':unittest.main()
