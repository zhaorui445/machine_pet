import unittest

from petlm.licensed_prose_pilot import contiguous_prose, render_prose, quality_reason, split_map, chapter


class LicensedProsePilotTests(unittest.TestCase):
    def test_adjacent_paragraphs_keep_context_without_joining_across_code(self):
        raw='# 标题\n\n自然语言用来表达我们的想法。\n\n这种表达需要结合上下文理解。\n\n```python\nprint(1)\n```\n\n代码后独立的说明段落。'
        rows=list(contiguous_prose(raw))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0][2],'自然语言用来表达我们的想法。\n\n这种表达需要结合上下文理解。')
        self.assertEqual(rows[1][2],'代码后独立的说明段落。')

    def test_images_formula_and_list_end_run_instead_of_leaving_holes(self):
        for excluded in ['![图](x.png)','计算结果是 $x=2$。','- 列表中的内容。','这个说明引用了 :cite:`a`。']:
            rows=list(contiguous_prose('完整的第一段。\n\n'+excluded+'\n\n完整的最后一段。'))
            self.assertEqual([r[2] for r in rows],['完整的第一段。','完整的最后一段。'])

    def test_chinese_wrapping_and_emphasis_render_without_word_gaps(self):
        self.assertEqual(render_prose('自然中文的**完整**句子,这里需要\n继续解释。'), '自然中文的完整句子,这里需要继续解释。')
        self.assertEqual(render_prose('这是*需要强调*的完整说明。'), '这是需要强调的完整说明。')

    def test_privacy_or_external_quote_blocks_are_not_retained(self):
        self.assertIsNone(render_prose('可以通过手机号13812345678联系作者。'))
        self.assertIsNone(render_prose('> 这是另一个作者的引用内容。'))

    def test_chapter_splits_independent_of_file_order_and_quality(self):
        paths=[f'chapter_{i}/a.md' for i in range(12)]+['chapter_0/b.md']
        a=split_map({'files':{'d2l':paths}})
        b=split_map({'files':{'d2l':list(reversed(paths))}})
        self.assertEqual(a,b)
        self.assertEqual(set(a.values()),{'train','validation','test'})
        self.assertEqual(chapter('d2l','chapter_0/a.md'),chapter('d2l','chapter_0/b.md'))
        self.assertEqual(chapter('hello_algo','docs/chapter_graph/graph.md'),'chapter_graph')

    def test_no_partition_fabricated_for_small_single_chapter_source(self):
        partitions=split_map({'files':{'cook':['tips/learn/a.md','tips/learn/b.md']}})
        self.assertEqual(set(partitions.values()),{'train'})

    def test_dependent_excerpt_requires_prior_context(self):
        self.assertEqual(quality_reason('因此我们可以继续。'*30),'missing_context')


if __name__=='__main__':
    unittest.main()
