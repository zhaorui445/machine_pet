import unittest

from petlm.dialogue_refine import REFERENCE_FIXES, REFERENCE_REJECTS, audit_pair


def source(quality=0.8,helpfulness=0.8,**labels):
    values={'quality':quality,'helpfulness':helpfulness,**labels}
    return {'labels':{key:{'value':value} for key,value in values.items()}}


class DialogueRefineTests(unittest.TestCase):
    def test_accepts_low_risk_high_quality_pair_with_missing_risk_labels(self):
        pair={'id':'1','prompt':'用户：请解释什么是惯性\n助手：',
              'answer':'惯性是物体保持原有运动状态的性质。'}
        self.assertEqual(audit_pair(pair,source())['review_status'],'approved')

    def test_rejects_high_stakes_and_explicit_risk(self):
        pair={'id':'2','prompt':'用户：头孢和酒能一起用吗\n助手：','answer':'可以。'}
        result=audit_pair(pair,source(pii=0.9))
        self.assertEqual(result['review_status'],'rejected')
        self.assertIn('high_stakes_or_time_sensitive',result['audit_reasons'])
        self.assertTrue(any(reason.startswith('explicit_risk:') for reason in result['audit_reasons']))

    def test_rejects_contextless_followup(self):
        pair={'id':'3','prompt':'用户：你更倾向于哪个看法？\n助手：','answer':'我倾向第一种。'}
        self.assertIn('missing_context',audit_pair(pair,source())['audit_reasons'])

    def test_reference_decisions_are_disjoint(self):
        self.assertFalse(set(REFERENCE_FIXES) & set(REFERENCE_REJECTS))
        self.assertEqual(len(REFERENCE_FIXES)+len(REFERENCE_REJECTS),28)


if __name__=='__main__':
    unittest.main()
