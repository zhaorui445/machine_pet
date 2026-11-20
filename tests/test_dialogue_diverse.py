import unittest

from petlm.dialogue_diverse import FINAL_FAMILIES, TRAIN_FAMILIES, final_blind_dialogue, training_dialogue


class DiverseDialogueTests(unittest.TestCase):
    def test_train_and_final_families_are_disjoint(self):
        self.assertFalse(set(TRAIN_FAMILIES)&set(FINAL_FAMILIES))

    def test_math_answers_are_calculated(self):
        item=training_dialogue('math',7,123)
        self.assertIn('乘以',item['answer'])
        blind,key=final_blind_dialogue('shopping_math',7,123)
        self.assertIn('总共',key['answer'])
        self.assertNotIn('answer',blind)

    def test_memory_correction_is_consistent(self):
        item=training_dialogue('memory',11,123)
        self.assertTrue(any(word in item['answer'] for word in ('更正','删除','不会把它保存')))


if __name__=='__main__':
    unittest.main()
