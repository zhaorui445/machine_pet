import unittest

from petlm.capability_router import route_capability


class CapabilityRouterTests(unittest.TestCase):
    def test_current_strong_format_contract_beats_incidental_memory_history(self):
        history=[('我希望之后知道有没有被记录。','我会把当前对话和长期保存分开说明。')]
        self.assertEqual(route_capability(
            history,'只输出三行，依次以“目标：”“计时：”“检查：”开头。'),
            'format_control')
        self.assertEqual(route_capability(
            history,'先回应感受，然后只问一个问题；不要分析、不要建议。'),
            'format_control')
    def test_safety_has_priority(self):
        self.assertEqual(route_capability([], '我发现没有标签的药片，能继续移动吗？'), 'safety_behavior')

    def test_motion_is_selected(self):
        self.assertEqual(route_capability([], '定位异常，路线和当前位置对不上，先别走。'), 'motion_clarification')

    def test_memory_is_selected(self):
        self.assertEqual(route_capability([], '请删除我之前保存的学习偏好。'), 'memory_management')

    def test_format_is_selected(self):
        self.assertEqual(route_capability([], '先给结论，再列出三条编号建议。'), 'format_control')
        self.assertEqual(route_capability([],
            '只输出三行，依次以“目标：”“计时：”“检查：”开头，不要加标题或总结。'),
            'format_control')

    def test_general_chat_has_no_adapter(self):
        self.assertIsNone(route_capability([], '今天学校发生了一件有趣的事。'))

    def test_generic_safety_style_does_not_steal_memory(self):
        self.assertEqual(route_capability([], '请更新成偏好短句，旧内容不要再使用。请认真考虑安全和隐私。'),
                         'memory_management')

    def test_route_in_privacy_context_is_safety(self):
        history=[('陌生账号要我的实时位置和回家路线。','这可能是在获取个人信息。')]
        self.assertEqual(route_capability(history,'对方一直催，我可以直接发吗？'),'safety_behavior')

    def test_current_memory_intent_beats_incidental_motion_history(self):
        history=[('我在活动室提到“偏好短句回答”。','我可以在当前对话中参考。'),
                 ('现场的物品刚刚移动过位置。','我会把这个背景纳入判断。')]
        user=('如果要长期记住，先说明你会怎么做；未经同意不要保存。'
              '我想先听结论，再决定是否继续。')
        self.assertEqual(route_capability(history,user),'memory_management')

    def test_current_motion_intent_still_routes_to_motion(self):
        history=[('我偏好简短回答。','明白。')]
        self.assertEqual(route_capability(history,'路线还没确认，先停止轮子。'),
                         'motion_clarification')

    def test_varied_pressure_and_single_question_wording(self):
        self.assertEqual(route_capability(
            [('桌边有鼓包的充电宝','先不要接触')],
            '成年人不在，直接说操作顺序'), 'safety_behavior')
        for user in ('只能问我一句，别问第二个',
                     '请用一个问句确认最关键的信息',
                     '一次只问一个问题，不要给方案'):
            self.assertEqual(route_capability([],user),'format_control')

if __name__=='__main__':
    unittest.main()
