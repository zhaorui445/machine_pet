# 中文对话训练实验

此页记录第一轮对话实验。后续扩充基础语料、清洗身份样本并再次训练的结果见 [SEMANTICS.md](SEMANTICS.md)；当前默认对话入口使用后续实验的检查点。

已从项目自己的基础模型出发完成一轮监督式对话训练。没有下载或使用其他模型权重，也没有调用云端聊天模型。

## 数据来源与处理

- 来源：[OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)，官方数据卡标注 Apache-2.0。使用固定仓库提交 `fdf72ae0827c1cda404aff25b6603abec9e3399b` 的 `2023-04-12_oasst_ready.messages.jsonl.gz`。
- 原始数据位于 `data/petlm/dialogue/raw/`，SHA256 为 `286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c`。
- 只保留中文、审核通过、未删除、非 synthetic、无 `model_name` 的人工用户与助手消息；不使用该来源中的其他模型生成内容。根据消息父节点重建最近几轮上下文，确保同一对话树只进入一个训练/验证/测试分组。
- 使用基础模型原有分词器，按 256 token 上下文限制筛选。最终得到训练 **940**、验证 **44**、测试 **64** 条问答，来自 **338** 个对话树。当前训练样本中有 283 条包含前文。
- 完整筛选计数、来源与局限见 `data/petlm/dialogue/prepared-oasst1-zh-context/manifest.json`。

这是通用人工问答，**没有经过电子宠物语气、儿童适龄性或事实准确性审核**。训练它只能让模型初步学习问答格式，不能把它称为成熟宠物模型。基础模型本身只有约 3148 万参数、约 619 万 token 的现代中文训练集，能力上限仍很低。

## 训练方式与结果

使用 `petlm/dialogue_train.py`，从 `data/petlm/models/curated-expanded/latest.pt` 加载权重，在单独目录保存新模型。只对**助手回复 token**计算损失；用户输入和上下文 token 不作为预测目标。优化器重新初始化，学习率 `5e-5`，训练 300 步，每 50 步评估并保留验证集最佳检查点。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_train --data data/petlm/dialogue/prepared-oasst1-zh-context --output data/petlm/models/dialogue-oasst1-zh-context --steps 300 --max-seconds 600
```

已完成全部 300 步。训练前验证集回复损失 **5.860**，训练后最佳为 **5.426**。独立测试集上，同一数据与分词器下基础模型为 **5.842**，新模型为 **5.559**。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_eval --checkpoint data/petlm/models/curated-expanded/latest.pt
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_eval --checkpoint data/petlm/models/dialogue-oasst1-zh-context/best.pt
```

测试损失下降代表回复 token 预测改善，**不代表回答已经连贯、正确或适龄**。实际试问“你好，今天我们做什么？”和“我有点不想学习”时，回复仍重复、语义断裂，并出现错误的身份表述。现阶段只适合研究演示，不能用作可靠学习或生活建议。

## 在 VS Code 试用

打开整个项目文件夹，在“运行和调试”中选 **电子宠物：本地对话实验**，按 F5；或在终端执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_chat
```

输入 `/exit` 退出。会话只保留在当前进程内，尚未接入长期记忆。单次提问：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.dialogue_chat --prompt "你好，今天我们做什么？"
```

当前训练数据不足且质量参差。要改善交流，下一步应收集并人工审核更多**面向电子宠物场景**的中文多轮对话，建立不依赖训练样本的固定测试题，并继续扩大、提升基础语言模型的语料与训练质量。不能靠重复训练同一千条问答保证流畅。
