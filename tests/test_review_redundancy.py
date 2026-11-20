import unittest

from petlm.reduce_review_redundancy import reduce_rows, similarity


def row(id, source, text, split='train'):
    return dict(id=id,title=source,split_group=source,split=split,text=text)


class ReviewRedundancyTests(unittest.TestCase):
    def test_distinct_text_same_source_caps_without_rewriting(self):
        a=row('a','书法','唐代书法的说明。')
        b=row('b','书法','清代书法及流派的介绍。')
        kept,removed=reduce_rows([a,b],preferred=['b'])
        self.assertEqual(kept,[b])
        self.assertEqual(removed[0]['reason'],'same_source_article_cap')

    def test_explicit_similar_description_cap_leaves_other_topics(self):
        a=row('a','鲸一','第一种鲸的体型特征。')
        b=row('b','鲸二','第二种鲸的外形说明。')
        c=row('c','猫','猫的食性描述。')
        kept,removed=reduce_rows([a,b,c],{'a':'鲸外形','b':'鲸外形'},['b'])
        self.assertEqual(kept,[b,c])
        self.assertEqual(removed[0]['reason'],'similar_topic_and_description_cap')

    def test_normalization_and_small_edits_are_detected(self):
        base='植物通过叶片进行光合作用，将阳光转化为自身生长所需的能量。根部从土壤中吸收水分，茎将水分向上输送。人们通过持续观察，可以了解植物在不同季节的变化，也可以比较不同生长条件产生的影响。'
        self.assertEqual(similarity('Ａ，中文。','A中文'),1.0)
        self.assertGreater(similarity(base,base.replace('人们','我们')), .75)
        a=row('a','甲',base)
        b=row('b','乙',base.replace('人们','我们'))
        kept,removed=reduce_rows([a,b])
        self.assertEqual(len(kept),1)
        self.assertEqual(removed[0]['reason'],'lexical_near_duplicate')

    def test_validation_precedence_preserves_original_partition(self):
        a=row('a','甲','同一段原文。')
        b=row('b','乙','同一段原文。','validation')
        kept,removed=reduce_rows([a,b],preferred=['a'])
        self.assertEqual(kept,[b])
        self.assertEqual(removed[0]['id'],'a')

    def test_duplicate_ids_fail_instead_of_overwriting_audit(self):
        a=row('a','甲','内容。')
        with self.assertRaisesRegex(ValueError,'Repeated'):
            reduce_rows([a,a])


if __name__=='__main__':
    unittest.main()
