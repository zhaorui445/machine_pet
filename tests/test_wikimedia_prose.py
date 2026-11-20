import bz2
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from petlm import wikimedia_prose as audit
from petlm import wikimedia_prose_select as selection
from petlm import author_prose


class WikimediaProseTests(unittest.TestCase):
    def test_mobile_encoded_parent_url_maps_to_same_book_group(self):
        a=selection.origin_group('https://zh.m.wikibooks.org/wiki/%E5%AD%B8%E7%BF%92/第一章')
        b=selection.origin_group('https://zh.wikibooks.org/w/index.php?title=学习/第二章&oldid=3')
        self.assertEqual(a,b)
        self.assertIsNone(selection.origin_group('https://other.example/wiki/学习'))

    def test_author_prose_does_not_splice_across_code_or_copy_quotes(self):
        paragraph='现代浏览器负责把网页内容呈现在用户面前,并且提供与页面交互的功能。开发者可以通过脚本描述页面的行为,让浏览器根据用户的操作更新内容。学习这些机制可以帮助我们理解网络应用的工作方式,也能帮助我们分析浏览器与服务器之间的关系。'
        md='# 入门\n\n'+paragraph+'\n\n```js\n'+paragraph+'\n```\n\n> '+paragraph+'\n\n![图](x.png)\n\n下面这个例子说明一个具体的浏览器行为。'+paragraph
        rows=list(author_prose.extract(md))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0][1],'入门')
        self.assertNotIn('下面',rows[0][2])
        self.assertEqual(list(author_prose.extract('- '+paragraph)),[])
        self.assertEqual(list(author_prose.extract('该方法的工作过程如下。'+paragraph)),[])

    def test_selection_rejects_observed_corruption_and_specialized_text(self):
        row={'source':'zhwikivoyage','title':'用电','text':'请确定确保电源没有任何问题。'}
        self.assertEqual(selection.classify(row)[1],'observed_defect_or_flattened_list')
        row={'source':'zhwikibooks','title':'初中语文/课文','text':'课文的自然文字。'}
        self.assertEqual(selection.classify(row)[1],'book_not_in_current_genre_selection')

    def test_chapters_and_script_variants_share_group(self):
        group = audit.group_key('zhwikibooks', '學習方法/预习')
        self.assertEqual(group, audit.group_key('zhwikibooks', '学习方法/复习'))
        self.assertNotEqual(group, audit.group_key('zhwikivoyage', '学习方法/复习'))

    def test_paragraphs_do_not_join_across_missing_formula_or_templates(self):
        raw = '== 学习 ==\n这是[[阅读|读书]]的完整段落。\n\n前句{{缺失的词}}后句。\n\n说明<math>x</math>结尾。\n\n独立的最后一段。'
        rows = list(audit.paragraphs(raw))
        valid = [row[2] for row in rows if row[3] is None]
        self.assertEqual(valid, ['这是读书的完整段落。', '独立的最后一段。'])
        self.assertEqual(rows[0][1], '学习')

    def test_paragraphs_remove_inline_citations_but_reject_other_html(self):
        from petlm.wikipedia_paragraph_candidates import paragraphs
        raw = '这是[[阅读|读书]]的完整段落<ref>{{cite web|title=x}}</ref>，同时也是说明。'
        rows = list(paragraphs(raw))
        self.assertEqual(rows[0][2], '这是读书的完整段落,同时也是说明。')
        self.assertIsNone(rows[0][3])
        bad = list(paragraphs('文字<br/>中断。'))
        self.assertIsNotNone(bad[0][3])

    def test_article_split_is_stable_and_page_grouped(self):
        from petlm.wikipedia_paragraph_candidates import page_split
        self.assertEqual(page_split('12345'), page_split('12345'))
        self.assertIn(page_split('12345'), ('train', 'validation', 'test'))

    def test_parent_sentence_overlap_and_heldout_precedence(self):
        shared = '这段话说明了应该如何安排学习时间，以及在复习过程中如何检查自己的理解，并根据实际情况调整计划。'
        a = {'id':'a','text':'训练开头。'+shared, 'split':'train'}
        b = {'id':'b','text':'验证开头。'+shared, 'split':'validation'}
        c = {'id':'c','text':'测试开头。完全不同的独立内容。', 'split':'test'}
        kept, rejected = audit.deduplicate([a, b, c], set())
        self.assertEqual({r['id'] for r in kept}, {'b','c'})
        self.assertEqual(rejected['candidate_text_prefix_paragraph_or_sentence_duplicate'], 1)
        kept, rejected = audit.deduplicate([a, b, c], audit.fingerprints('父模型不同的开头。'+shared))
        self.assertEqual([r['id'] for r in kept], ['c'])
        self.assertEqual(rejected['parent_text_prefix_paragraph_or_sentence_overlap'], 2)

    def test_xml_keeps_only_articles_without_contributor_profiles(self):
        xml = '''<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"><siteinfo/>
        <page><title>教材/第一章</title><ns>0</ns><id>1</id><revision><id>10</id><timestamp>2026</timestamp><contributor><username>private-profile</username></contributor><text>正文</text></revision></page>
        <page><title>讨论:教材</title><ns>1</ns><revision><text>不收录</text></revision></page>
        <page><title>重定向</title><ns>0</ns><redirect title="教材"/><revision><text>不收录</text></revision></page></mediawiki>'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source.bz2'
            path.write_bytes(bz2.compress(xml.encode()))
            rows = list(audit.page_rows(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['revision_id'], '10')
            self.assertNotIn('private-profile', json.dumps(rows))

    def test_changed_evidence_is_rejected_before_other_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'dumps-legal.html').write_bytes(b'changed')
            (root/'dumps-legal.html.receipt.json').write_text(json.dumps({
                'bytes':3, 'sha256':hashlib.sha256(b'old').hexdigest()}))
            with self.assertRaisesRegex(ValueError, 'Changed source'):
                audit.verify_source(root)

    def test_privacy_risk_context_and_markup_are_rejected(self):
        prose = '这是一个需要认真检查的完整说明段落。'*15
        for payload, expected in [('手机13812345678。', 'personal_or_contact_pattern'),
                                  ('治疗建议。', 'outside_low_risk_prose_scope'),
                                  ('残留{{模板}}。', 'residual_markup_or_defect')]:
            self.assertEqual(audit.reason(prose+payload), expected)


if __name__ == '__main__':
    unittest.main()
