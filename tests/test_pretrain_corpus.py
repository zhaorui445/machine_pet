import json
from pathlib import Path
import tempfile
import unittest

from petlm.pretrain_corpus import (FINEWEB_SPAM_ORIGIN, clean_document,
                                   dialogue_rows, repeated_shingle_ratio,
                                   to_simplified)


class PretrainCorpusTests(unittest.TestCase):
    def test_wikipedia_tail_and_duplicate_lines_are_removed(self):
        unique=(
            '地球围绕太阳运行，完成一次公转大约需要一年。',
            '植物的根能固定植株，也会从土壤中吸收水和无机盐。',
            '液态水在标准大气压下冷却到零摄氏度附近会结冰。',
            '鸟类的羽毛具有保温作用，不同物种的飞行能力差异很大。',
            '汉字记录汉语已有很长历史，现代书写使用规范化字形。',
            '计算机通过处理指令和数据完成运算，程序描述具体步骤。',
            '月球是地球的天然卫星，它的引力会影响海洋潮汐。',
            '声音由物体振动产生，需要通过介质传播到人的耳朵。',
            '地图使用符号表示地形、道路和其他空间信息。',
            '人体需要均衡饮食和适量运动，健康情况存在个体差异。',
            '历史研究需要比较多种来源，并区分事实、解释和推测。',
            '科学实验应记录条件和结果，危险操作需要专业人员指导。',
        )
        lines=[]
        for line in unique:
            lines.extend((line,line))
        text='\n'.join(lines)+'\n参考资料\n无关链接内容。'
        cleaned,reason,signals=clean_document(text,'wikipedia')
        self.assertIsNone(reason)
        self.assertNotIn('参考资料',cleaned)
        self.assertGreater(signals['duplicate_lines_removed'],0)

    def test_unexpected_scripts_and_spam_are_rejected(self):
        self.assertEqual(clean_document('送彩金'+('这是中文句子。'*100),'fineweb2')[1],'spam_pattern')
        converted_spam=('這是普通網頁內容。'*40)+'代開發票，廣告合作請聯繫QQ。'
        self.assertEqual(clean_document(converted_spam,'fineweb2')[1],'spam_pattern')
        self.assertEqual(clean_document(('这是中文句子。'*100)+'한국어','wikipedia')[1],
                         'unexpected_script_or_symbols')

    def test_fineweb_gambling_and_diploma_spam_are_rejected(self):
        gambling='影视简介和节目资料。'+'百家乐群和娱乐城盘口变化。'*30
        diploma='游戏故事内容。'+'买真毕业证，可在学信网永久查询。'*30
        self.assertEqual(clean_document(gambling,'fineweb2')[1],'fineweb_spam_pattern')
        self.assertEqual(clean_document(diploma,'fineweb2')[1],'fineweb_spam_pattern')
        traditional='這是網頁內容。'+'線上老虎機與百家樂娛樂城。'*30
        self.assertEqual(clean_document(traditional,'fineweb2')[1],'fineweb_spam_pattern')
        self.assertIsNotNone(FINEWEB_SPAM_ORIGIN.search('http://y12bet.example/a.html'))
        promotion=('医院皮肤科提供诊疗服务。专家扣扣：923039999，地址在某街道。')*10
        self.assertEqual(clean_document(promotion,'fineweb2')[1],
                         'fineweb_promotional_contact')
        legitimate=(
            '赌博是一种以财物为赌注的活动，历史上存在多种形式。'
            '许多地区通过法律限制经营场所、参与年龄和广告方式。'
            '公共卫生研究会调查成瘾风险及其对家庭生活的影响。'
            '经济学者分析税收、监管成本和非法市场之间的关系。'
            '社会工作机构可能提供债务咨询与心理支持服务。'
            '不同司法管辖区的定义、许可制度和处罚规则并不相同。'
            '历史资料显示，相关政策会随社会观念和技术变化而调整。'
            '研究结论需要说明样本、方法和适用范围，不能任意推广。'
            '百科条目应描述概念和历史，而不是提供参与活动的方法。')
        self.assertIsNone(clean_document(legitimate,'wikipedia')[1])

    def test_repeated_shingle_ratio_detects_copy_loops(self):
        self.assertGreater(repeated_shingle_ratio('欢迎关注本站。'*200),0.22)

    def test_traditional_conversion_reaches_a_fixed_point(self):
        converted=to_simplified('隶属於机构，乾纲独断，资料过於单一。')
        self.assertEqual(to_simplified(converted),converted)

    def test_html_cleanup_does_not_cross_line_boundaries(self):
        paragraphs=[
            '科学观察需要记录对象、条件和变化，结论应与证据相符。',
            '学习新概念时可以先提出问题，再查找资料并核对来源。',
            '地图上的比例尺用于表示图上距离与实际距离之间的关系。',
            '植物利用光能进行光合作用，并在生态系统中承担重要角色。',
            '安全实验应遵守操作规范，使用合适器材并记录异常情况。',
            '历史解释需要区分史料内容、研究者判断和仍待验证的推测。',
        ]
        raw='孤立的小于号<\n'+'\n'.join(paragraphs*3)+'\n末尾的大于号>'
        cleaned,reason,_=clean_document(raw,'wikipedia')
        self.assertIsNone(reason)
        self.assertIn('科学观察需要记录对象、条件和变化',cleaned)
        self.assertEqual(clean_document(cleaned,'wikipedia')[0],cleaned)

    def test_nested_wiki_templates_are_removed_to_fixed_point(self):
        paragraphs=(
            '地球围绕太阳运行，轨道和自转共同影响季节与昼夜变化。',
            '河流携带泥沙向下游移动，并持续塑造河谷与三角洲地貌。',
            '人体通过呼吸系统交换气体，运动时氧气需求通常会增加。',
            '阅读历史资料要核对作者、年代、目的以及保存过程。',
            '程序设计把复杂任务拆成步骤，再通过测试发现和修正错误。',
            '植物根系吸收水和无机盐，叶片则利用光能合成有机物。',
            '声音来自物体振动，传播速度会受到介质性质与温度影响。',
            '地图比例尺连接图上距离与实际距离，方向和图例也很重要。',
            '均衡饮食需要考虑食物种类、数量以及个人身体活动水平。',
            '科学结论必须说明证据与适用条件，并允许后续研究修正。',
        )
        raw='{{外层模板|{{内层模板|参数}}}}\n'+'\n'.join(paragraphs)
        cleaned,reason,_=clean_document(raw,'wikipedia')
        self.assertIsNone(reason)
        self.assertNotIn('模板',cleaned)
        self.assertEqual(clean_document(cleaned,'wikipedia')[0],cleaned)

    def test_character_run_normalization_cannot_create_late_duplicates(self):
        unique=(
            '地球围绕太阳运行，轨道和自转共同影响季节与昼夜变化。',
            '河流携带泥沙向下游移动，并持续塑造河谷与三角洲地貌。',
            '人体通过呼吸系统交换气体，运动时氧气需求通常会增加。',
            '阅读历史资料要核对作者、年代、目的以及保存过程。',
            '程序设计把复杂任务拆成步骤，再通过测试发现和修正错误。',
            '植物根系吸收水和无机盐，叶片则利用光能合成有机物。',
            '声音来自物体振动，传播速度会受到介质性质与温度影响。',
            '地图比例尺连接图上距离与实际距离，方向和图例也很重要。',
            '均衡饮食需要考虑食物种类、数量以及个人身体活动水平。',
            '科学结论必须说明证据与适用条件，并允许后续研究修正。',
            '团队合作要明确目标和分工，遇到分歧时应依据事实讨论。',
            '学习新概念可以先举实例，再总结规律并用练习检验理解。',
        )
        raw='\n'.join(unique+('halo........','halo.........'))
        cleaned,reason,signals=clean_document(raw,'fineweb2')
        self.assertIsNone(reason)
        self.assertEqual(cleaned.count('halo..'),1)
        self.assertGreater(signals['duplicate_lines_removed'],0)
        self.assertEqual(clean_document(cleaned,'fineweb2')[0],cleaned)

    def test_dialogue_rows_use_manifest_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'manifest.json').write_text(json.dumps({'source':'OpenAssistant/oasst2',
                'revision':'fixed-revision','license':'Apache-2.0'}),encoding='utf-8')
            path=root/'train.jsonl'
            path.write_text(json.dumps({'id':'row-1','split':'train','prompt':'用户：你好\n助手：',
                'answer':'你好呀。'},ensure_ascii=False)+'\n',encoding='utf-8')
            row=next(dialogue_rows(path))
        self.assertEqual(row['source'],'openassistant_oasst2')
        self.assertEqual(row['dataset'],'OpenAssistant/oasst2')
        self.assertIn('fixed-revision',row['origin'])

    def test_dialogue_rows_support_globally_deduplicated_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'manifest.json').write_text(json.dumps({
                'kind':'human_authored_chinese_dialogue_merge'}),encoding='utf-8')
            path=root/'train.jsonl'
            path.write_text(json.dumps({'id':'row-2','split':'train','prompt':'用户：你好\n助手：',
                'answer':'你好呀。','source_dataset':'OpenAssistant/oasst1',
                'source_revision':'merged-revision','source_license':'Apache-2.0'},
                ensure_ascii=False)+'\n',encoding='utf-8')
            row=next(dialogue_rows(path))
        self.assertEqual(row['source'],'openassistant_oasst1')
        self.assertIn('merged-revision',row['origin'])


if __name__=='__main__':
    unittest.main()
