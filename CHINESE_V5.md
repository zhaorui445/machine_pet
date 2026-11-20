# 中文模型 v5 训练与验证记录（2026-11-03）

本轮已完成中文语料重筛、全新分词器、随机权重基础训练、对话训练和应用层保护。模型没有接入任何现成模型权重或云端聊天服务。它仍是研究原型，不能用于儿童独立陪伴、安全决策、医疗建议或真实机器人控制。

## 当前默认版本

- 基础模型：`data/petlm/models/chinese-v5-b32/latest.pt`
- 对话模型：`data/petlm/models/dialogue-diverse-v6/best.pt`
- 分词器：`data/petlm/tokens-chinese-v5/tokenizer.model`
- 对话数据：`data/petlm/dialogue/prepared-pet-chinese-v5/`
- 固定题结果：`data/petlm/evaluation/dialogue-chinese-v5-final-probes-19.json`

`petlm.dialogue_chat` 已默认使用上述对话模型和 v6 复用的16K分词器，生成温度默认为 0.3。早期检查点仍保留用于对照。

## 数据与分词器

本机原始数据重筛后保留 6955 篇：FineWeb2 中文 3067 篇、中文维基百科 3888 篇；划分为训练 6264 篇、验证 326 篇、测试 365 篇。训练部分共有 **12,009,389 个唯一语料 token**。因 Hugging Face 网络连接持续超时，本轮没有取得更多 FineWeb2 数据；CLUECorpus2020 和中文维基文库也没有进入这个快照。

筛选要求中文比例至少 55%，并过滤韩文、日文假名、异常字符、重复短片段和明显垃圾文本。新 SentencePiece BPE 分词器有 16000 个词项，启用 byte fallback；测试集未知 token 比例为 0。生成时会屏蔽 `<unk>`、BOS 和 256 个 byte token，避免把 byte 片段直接输出给用户。

“唯一语料规模”和“模型累计处理量”是两个不同指标：模型用 12,009,389 个训练 token 重复采样训练 12500 步，每步处理 8192 token，因此累计处理量是 **102,400,000 token**，不能表述为拥有 1 亿条独立训练 token。

## 模型与训练结果

基础模型包含 33,526,784 个参数，结构为 8 层、宽度 512、8 个注意力头、上下文 256，批量 32，从随机权重训练 12500 步。最终验证损失为 4.789；重新计算的测试损失为 4.744，困惑度为 114.913。损失下降只能说明留出文本预测有所改善，不能证明回答正确。

对话快照包含训练 796、验证 29、测试 56 组对话，其中 30 组为用户确认过的电子宠物对话。训练只计算助手回复损失。第 150 步验证损失最低，为 5.034；后续连续退化，因此在第 350 步提前停止。相同 56 组测试问答上，基础模型回复损失为 4.888，对话模型为 4.875，改善很小。

## 语义和保护层

神经网络本体仍会答非所问、重复中文或生成无意义的拉丁字母拼接。继续在同一批少量对话上训练更容易过拟合，不能从根本上解决这一问题。

本轮自由问题抽测也证实了这一点：“为什么天空看起来是蓝色的”触发了保守回复；“水为什么会结冰”生成了高频重复“水”的错误内容，随后已加入表面重复拦截；“我今天在公园看见很多鸟”仍给出答非所问但形式正常的中文。最后一种错误无法靠字符规则可靠识别，必须依赖更多高质量训练数据和独立语义评测。

当前聊天入口在模型外增加了三类确定性处理：简单分数和乘法；提醒、移动、视觉等能力边界；用药、易燃实验、隐私、欺凌等安全边界。还加入了经审核的自我介绍、年龄语气、学习和情绪支持等常见宠物场景。无法通过中文比例、异常脚本和重复检测的模型输出会被替换为“不理解”的保守回复。

固定集目前有 19 题。评估 JSON 的 `response_source` 会区分：

- `deterministic_rule`：应用层审核规则直接回答，不代表模型理解；
- `model_generation`：神经网络实际生成；
- `generation_fallback`：模型输出未通过质量检查，已使用保守回复。

本轮19题都命中了确定性规则，所以只能证明这些已知场景的应用层行为稳定，不能据此声称“模型语义已经解决”。

## 运行与复核

在 VS Code 中选择“电子宠物：本地对话实验”并按 F5，或在项目终端运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_chat
```

单题测试和固定集复核：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_chat --prompt "你好，你是谁？"
.\.venv\Scripts\python.exe -X utf8 -m petlm.probe --checkpoint data/petlm/models/dialogue-chinese-v5/best.pt --tokenizer data/petlm/tokens-chinese-v5/tokenizer.model --temperature 0.3 --output data/petlm/evaluation/dialogue-chinese-v5-final-probes-19.json
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

2026-11-03 已生成2万条多轮候选和1000题隔离盲测集，用户随后确认整批数据合格。确认方式被记录为整批确认而非逐条审核。新快照 `data/petlm/dialogue/prepared-pet-expanded-v2/` 含训练20796条、验证29条、测试56条，已可用于下一轮训练；盲测答案保持隔离。审核和评测流程见 [DIALOGUE_EXPANSION.md](DIALOGUE_EXPANSION.md)。

扩充训练随后完成。最终用于训练的 v3 快照单独抽取2%新增数据作内部验证。保守模型在1000题启发式盲测中从旧模型的1.0%提高到10.7%，但大部分提升集中于情绪倾听，多个关键场景仍为0%，旧56题损失也从4.875退化到4.992。因此新检查点已保存用于实验，但没有替换默认模型。详细结果见 [DIALOGUE_EXPANSION.md](DIALOGUE_EXPANSION.md)。

进一步的 v5 改进改用4000条更丰富的合成对话和30%旧数据回放。新检查点 `data/petlm/models/dialogue-diverse-v5/best.pt` 在旧56题上的损失为4.886，基本保持原能力，并已成为默认检查点。新的独立500题上，模型本体只通过7题；加入应用层计算和安全边界后，开发评估为418题，但其中420题由规则直接回答。模型本体与完整应用的能力必须继续分开报告。
