# 语义改进实验（2026-11-02）

> 本页记录 2026-11-02 的旧实验。2026-11-03 从随机权重重新训练的中文 v5 版本、默认检查点和最新限制见 [CHINESE_V5.md](CHINESE_V5.md)。

用户反馈：旧对话模型会自称其他模型、重复词句，并在简单问题上答非所问。本轮仍从项目自己的权重继续训练，没有接入现成模型权重或云端聊天服务。**语义问题尚未解决**，本检查点仅供本地研究演示。

## 做了什么

- 从维基百科官方转储追加 8000 篇候选文章，导入后新增 7568 篇。对已有 FineWeb2 与维基百科文本重新筛选，保留 9289 篇，其中训练集约 2460 万 token。快照为 `data/petlm/prepared-curated-wiki-expanded/`，分词数据为 `data/petlm/tokens-curated-wiki-expanded/`；沿用原分词器以便继续使用项目权重。此快照没有 CLUECorpus2020 或维基文库。
- 在 3148 万参数的基础模型上，用新语料训练 5000 步，再以较低学习率继续到 10000 步。两个新阶段分别保存在 `data/petlm/models/language-v2/` 和 `data/petlm/models/language-v2-extension/`，旧检查点保留。
- 从 OpenAssistant/oasst1 人工中文问答中去除提及其他模型名称及第一人称 AI 身份的助手回答。首次快照 `data/petlm/dialogue/prepared-oasst1-zh-persona-clean/` 含训练 786、验证 34、测试 55 条。复核发现身份描述还存在于少数助手前文和不同措辞中，于是扩大过滤并生成 `data/petlm/dialogue/prepared-oasst1-zh-persona-clean-v2/`，含训练 717、验证 26、测试 50 条。来源、许可、原始文件哈希和筛选计数见各自的 `manifest.json`。
- 从第 5000 步和第 10000 步基础模型启动对话训练。只对助手回复计算损失，选验证集最优检查点。最终默认使用清洗更彻底的 `data/petlm/models/dialogue-v3/best.pt`（第 200 个对话训练步）。

## 实测结果

下表的回复损失都在**首次清洗快照的同一批 55 条留出问答**上计算，数值越低越好；它不衡量语义连贯、事实正确或儿童适龄性。

| 检查点 | 留出回复损失 |
| --- | ---: |
| 旧基础模型 | 5.866 |
| 旧对话模型 | 5.677 |
| 新基础模型，5000 步 | 5.112 |
| 新对话模型，5000 步基础 + 250 步对话 | 5.001 |
| 新基础模型，10000 步 | 4.879 |
| 新对话模型，10000 步基础 + 150 步对话 | **4.765** |

在第二次清洗快照的**同一批 50 条留出问答**上，基础模型、首次清洗对话模型、再次清洗对话模型的损失分别为 4.881、4.792、4.835。再次清洗模型在相同 26 条验证问答上的损失是 5.016，首次清洗模型是 5.012；清洗更彻底并没有带来可测的语义提升。两次快照的分数不能直接相互比较。

基础模型在新文本验证集上的损失从第 5000 步的 4.858 降至第 10000 步的 4.634。固定问题的输出保存在 `data/petlm/evaluation/`。例如最终默认模型被问“二分之一加三分之一等于多少？”时回答“因为你可以将任何问题从中提高”，被问学习建议时也答非所问。因此**不能用于学习辅导、提醒承诺或其他需要准确回答的场景**。身份关键词过滤只能减少训练样本中的误导，不能保证生成时始终保持角色。

## 扩展对话检查

`petlm.probe` 现在包含 18 条固定题，覆盖自我介绍、两种年龄语气、学习计划、分数计算、纠错、情绪支持、记忆、提醒、移动和视觉能力、隐私、欺凌、健康、危险实验、不确定事实、简洁回答和笑话。对话模型用默认温度 0.7 和较低温度 0.3 分别测试；基础模型也用温度 0.7 测试。输出分别保存在 `dialogue-pet-reviewed-v1-probes-18.json`、`dialogue-pet-reviewed-v1-probes-18-t03.json` 和 `language-v2-extension-probes-18.json`。

人工检查结论：18 条中没有一条能同时做到切题且可靠。低温度让措辞略短，但没有修复理解和事实问题；对酒精实验的拒绝、隐私保护、用药边界和楼梯感知尤其不可靠。模型不得实际用于儿童陪伴、安全决策或学习指导。这是定性小测，不是盲评或正式基准。

同一份留出对话集验证损失：基础模型 5.206，对话训练模型 5.032；同一测试集损失：基础模型 4.881，对话训练模型 4.821。两者都有小幅改善，但生成检查说明损失下降没有转化成可靠回答。

## 在 VS Code 试用

打开整个项目文件夹，进入“运行和调试”，选择 **电子宠物：本地对话实验**，按 F5。也可以在项目终端运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_chat
```

输入 `/exit` 结束。若要指定某个检查点：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_chat --checkpoint data/petlm/models/dialogue-v3/best.pt --prompt "你好，今天我们做什么？"
```

复核留出集和固定问题：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_eval --checkpoint data/petlm/models/dialogue-v3/best.pt --data data/petlm/dialogue/prepared-oasst1-zh-persona-clean-v2
.\.venv\Scripts\python.exe -X utf8 -m petlm.probe --checkpoint data/petlm/models/dialogue-v3/best.pt
```

下一步应先取得更大、更干净且有使用权的现代中文语料，以及经过人工审核的电子宠物多轮对话；同时建立人工评分题，覆盖连贯性、身份、数学、记忆与不能执行的提醒。当前约 2460 万 token 的基础训练集和不到 800 条对话训练样本，尚不足以让从零训练的模型稳定交流。继续在同一小批对话上重复训练可能过拟合，不能当作语义修复。

## 新增宠物对话草稿

`data/petlm/dialogue/pet-scenarios-review.jsonl` 有 30 条中文宠物场景对话，覆盖学习陪伴、基础数学、年龄语气、情绪支持、隐私与安全、记忆和硬件能力边界等主题。用户确认内容无误后，30 条都登记为通过，审核者字段记为“用户确认”，日期为 2026-11-02。审核中我还加强了欺凌和离家出走两条的求助指引。

条目仍可在 VS Code 中复查；每行保留审核状态、审核者和日期。人工审核重点包括数学结果、健康建议、欺凌求助、隐私、安全拒绝，以及提醒和实体移动的能力边界。

审核合并已完成。以下命令供复现；若再次运行，请把 `--output` 改成新的目录名：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_curate prepare --drafts data/petlm/dialogue/pet-scenarios-review.jsonl --base data/petlm/dialogue/prepared-oasst1-zh-persona-clean-v2 --output data/petlm/dialogue/prepared-pet-reviewed-v1 --tokenizer data/petlm/tokens-curated-wiki-expanded/tokenizer.model
```

合并结果为训练 747、验证 26、测试 50 条。合并器要求每条通过样本都有审核者、日期、唯一 ID 和非空问答；待审核条目会被跳过。结果清单记录草稿文件哈希和审核状态计数。对话训练已完成：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_train --data data/petlm/dialogue/prepared-pet-reviewed-v1 --tokenizer data/petlm/tokens-curated-wiki-expanded/tokenizer.model --parent data/petlm/models/language-v2-extension/latest.pt --output data/petlm/models/dialogue-pet-reviewed-v1 --steps 300 --max-seconds 600
```

训练保存在新目录 `data/petlm/models/dialogue-pet-reviewed-v1/`，第 150 步验证集最佳回复损失为 5.032。相同 50 条留出问答上，基础模型损失为 4.881，新对话模型为 4.821。固定问题输出保存在 `data/petlm/evaluation/dialogue-pet-reviewed-v1-probes.json`；答案仍然明显不连贯，所以当前 VS Code 默认入口继续使用此前检查点。新模型可显式试用：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_chat --checkpoint data/petlm/models/dialogue-pet-reviewed-v1/best.pt
```

审核后的 30 条样本确已纳入训练，但这批数据规模仍小，损失的轻微下降没有转化为稳定交流能力。
