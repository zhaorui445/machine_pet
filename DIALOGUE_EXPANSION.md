# 多轮对话扩充与盲测流程

## 当前产物

正式候选版本为 `data/petlm/dialogue/expansion-v2/`：

- `candidates.pending.jsonl`：20000 条多轮训练候选，12 个主题，每个主题约 1667 条；
- `blind/prompts.jsonl`：1000 条无答案盲测输入；
- `blind/answer-key.pending.jsonl`：与盲测输入分开的参考回答和评分要点；
- `manifest.json`：数量、哈希、模板隔离和分词长度检查。

训练候选和盲测使用不同模板族、不同 ID，且没有相同的规范化提示。全部样本适配中文 v5 的 256 token 上下文，最长提示加回答为 168 token。

用户已于2026-11-03确认整批数据合格。确认记录明确标为 `user_batch_confirmation` 和 `corpus_level_not_item_by_item`，因此可以描述为“用户整批确认”，不能描述为“2万条逐条人工审核”。训练确认快照为 `reviewed-batch-confirmed.jsonl`，盲测参考确认快照为 `blind/answer-key.batch-confirmed.jsonl`。`expansion-v1` 是发现模板一致性问题前的首次生成结果，已由 v2 取代，不应训练使用。

已建立可训练对话快照 `data/petlm/dialogue/prepared-pet-expanded-v2/`：训练20796条、验证29条、测试56条。其中新增20000条来自本次整批确认，盲测内容没有加入该快照。

## 在 VS Code 审核

若后续要在整批确认基础上继续做更严格的逐条抽查，可打开“终端 -> 运行任务”，选择“电子宠物：审核下一批对话”，输入审核人姓名或代号。每次默认审核25条：

- `a`：通过；
- `r`：拒绝；
- `e`：修改回答后通过；
- `s`：暂时跳过；
- `q`：保存当前进度并退出。

逐条抽查决定写入 `data/petlm/dialogue/expansion-v2/review-decisions.jsonl`，不会覆盖整批确认记录；原候选文件也保持不变。选择“电子宠物：查看对话审核进度”可查看逐条抽查数量和剩余主题。

也可以在终端运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_expand review --source data/petlm/dialogue/expansion-v2/candidates.pending.jsonl --decisions data/petlm/dialogue/expansion-v2/review-decisions.jsonl --reviewer "审核人代号" --limit 25
```

逐条检查：上下文与回答是否对应、事实是否正确、年龄是否合适、是否泄露隐私、危险场景是否明确拒绝、能力边界是否真实、措辞是否自然。2万条逐条审核是数百小时规模的工作，适合多名审核者分批进行；当前整批确认不会被冒充为逐条人工审核。

## 导出已审核数据

先把决定合并为审核快照：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_expand materialize --source data/petlm/dialogue/expansion-v2/candidates.pending.jsonl --decisions data/petlm/dialogue/expansion-v2/review-decisions.jsonl --output data/petlm/dialogue/expansion-v2/reviewed-v1.jsonl
```

`reviewed-v1.jsonl` 中未审核条目仍为 `pending`，现有 `petlm.dialogue_curate` 只会导入明确 `approved` 且带审核人和日期的条目。达到计划审核量后，再用新输出目录建立训练快照，不覆盖旧快照。

## 独立盲测

盲测参考回答已经整批确认并保存在 `blind/answer-key.batch-confirmed.jsonl`。该文件绝不能传给训练程序。模型运行时只读取不含答案的 `blind/prompts.jsonl`：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.blind_eval --mode model --output data/petlm/evaluation/chinese-v5-blind-model.jsonl
```

`--mode model` 绕过确定性保护层，测神经网络本体；`--mode application` 测包含安全规则和回退机制的完整聊天入口。输出只包含模型回答，不读取参考答案。正式评分应由没有参与训练数据制作的审核者依据单独评分要点完成，并同时报告各主题通过率、严重安全错误数和保守回退率。

已用 `--mode model --limit 10` 做过不读取答案的冒烟测试，结果保存在 `data/petlm/evaluation/chinese-v5-blind-model-smoke-10.jsonl`。人工查看发现10题全部答非所问，涉及学习、情绪、安全、隐私、移动、记忆、语气、数学、游戏规则和不确定性。这不是完整1000题的正式评分，但足以确认当前模型本体尚不具备可靠多轮对话能力。

## 扩充数据训练结果

为避免新增模板全部进入训练而没有代表性验证集，最终训练快照使用 `data/petlm/dialogue/prepared-pet-expanded-v3/`：训练20408条、验证417条、测试56条。新增确认数据按 ID 哈希固定抽取2%进入内部验证；1000题外部盲测仍完全隔离。

第一轮从 `dialogue-chinese-v5/best.pt` 继续训练，学习率3e-5、有效批量16，计划2000步。第400步验证损失最低0.691，第900步因连续5次退化自动停止。检查点保存在 `data/petlm/models/dialogue-expanded-v3/`。它在旧56题上的回复损失为5.048，比训练前4.875更差。

第二轮采用更保守的学习率1e-5并训练400步，最佳验证损失0.705，检查点保存在 `data/petlm/models/dialogue-expanded-v3-conservative/`。旧56题损失为4.992，仍差于训练前，但好于第一轮。

三个模型都在相同1000题上以 `--mode model` 运行。保守关键词与数值检查结果如下；这不是正式人工语义评分：

| 模型 | 最低标准通过 | 主要表现 |
| --- | ---: | --- |
| 原对话模型 | 10/1000（1.0%） | 各场景基本失败 |
| 第一轮扩充训练 | 102/1000（10.2%） | 98题集中在情绪倾听；数学1题 |
| 保守扩充训练 | 107/1000（10.7%） | 情绪倾听96题；危险实验拒绝7题；其他多数场景仍失败 |

盲测报告分别为 `dialogue-chinese-v5-blind-score.json`、`dialogue-expanded-v3-blind-score.json` 和 `dialogue-expanded-v3-conservative-blind-score.json`。扩充训练明显学习了部分模板措辞，但没有形成可靠的跨措辞泛化，并伴随旧问答能力下降。因此默认聊天入口没有切换到新检查点。

显式试用保守模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_chat --checkpoint data/petlm/models/dialogue-expanded-v3-conservative/best.pt --prompt "我一看到整页任务就想放弃，该怎么开始？"
```

## 多样化 v5 改进

上一轮2万条数据虽然数量大，但自然表达组合有限。后续改为4000条真正唯一的组合式增强数据，每个目标场景400条，并重新从原始796条对话构建 `data/petlm/dialogue/prepared-pet-diverse-v5/`。训练集4709条，其中旧对话796条、多样化增强3913条；训练采样固定30%回放旧对话。增强数据标记为程序化检查的合成数据，未冒充人工逐条审核。

新模型保存在 `data/petlm/models/dialogue-diverse-v5/best.pt`。计划1200步，第650步验证损失最低2.655，第900步早停。旧56题回复损失为4.886，接近原模型4.875，明显好于上一轮模板训练的4.992，说明回放策略基本抑制了遗忘。

新的500题盲测位于 `data/petlm/dialogue/final-blind-v3/`，训练和盲测模板族、ID及完全相同提示均隔离。首次模型本体评估时，旧模型通过0/500，多样化 v5 通过7/500（1.4%）。这说明神经网络仍不具备可靠的跨措辞泛化，不能仅靠当前规模的合成对话解决基础语言能力不足。

应用层随后增加了文字数学、危险材料、证件隐私、传感器离线、记忆撤回、回答格式和游戏规则处理，并支持在有限条件下读取上一轮用户上下文。应用模式开发评估由254/500提高到418/500（83.6%）。由于这些规则是在查看同一套500题的失败后调整的，418/500是**开发集分数，不是独立最终成绩**。其中420题由确定性规则回答，不能归功于神经网络语义理解。

这一阶段曾将默认聊天检查点切换为 `dialogue-diverse-v5/best.pt`；后续已由下文的 v6 取代。温度0的贪心解码只有4/500，低于采样结果，因此没有改为默认。

## 中文预训练 v6

本地14,117篇原始文档重新做质量筛选，构建 `data/petlm/prepared-chinese-v6/`。v6 保留5734篇维基百科、4036篇 FineWeb2 中文和766条仅来自 OpenAssistant 训练分区的人类中文对话，共10536篇；验证和测试对话未进入预训练。处理过程中删除660个重复行，截去4376个维基参考资料尾部，并继续过滤垃圾、异常脚本、过多链接和高重复文本。

使用原16K分词器后，v6 训练集包含19,743,411个唯一语料 token，比 v5 的12,009,389增加64%。基础模型从 v5 权重以3e-5学习率适配3000步，累计处理24,576,000 token。固定留出窗口上，验证损失从4.929降至4.829，测试损失从4.861降至4.784，测试困惑度从129.15降至119.63。

从 v6 基础权重重新进行多样化对话训练后，第700步最佳验证回复损失为2.577，优于 v5 的2.655；旧56题损失由4.886降至4.687。500题模型本体最低标准由7题提高到9题，仍然很低；完整应用因规则覆盖保持418/500，其中420题由确定性规则回答。

因此默认模型已切换到 `data/petlm/models/dialogue-diverse-v6/best.pt`。新增 FineWeb2 10000篇的官方接口下载再次因 `ConnectTimeout [WinError 10060]` 失败，未将空 partial 文件导入。v6 证明更多高质量预训练有效，但19.7M唯一 token 仍不足以形成可靠自由对话。
