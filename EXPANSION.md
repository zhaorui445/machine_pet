# 语料扩充与继续训练记录

本轮工作基于已有三源模型，扩充并筛选语料，再从其 1500 步检查点继续训练。它仍是**基础文本续写模型**，没有完成宠物对话训练。

## 数据

- FineWeb2 官方 `cmn_Hani` 接口：在原有 1000 篇之外，下载了行偏移 1000–5999 的 5000 篇样本。下载曾因 429 限流中断，随后从已保存的 4000 篇继续完成。原始文件与 SHA256 记录在 `data/petlm/raw/`。
- 原有维基百科与维基文库资料保持不变。CLUECorpus2020 仍需向官方申请，未使用。
- 新快照位于 `data/petlm/prepared-curated-expanded/`。先从已有 SQLite 文档库筛选，去除明显广告、过短或低中文比例文本、过多链接、明显重复块和导航文本；用 OpenCC 将繁体字转为简体字，按标准化正文及相同开头去重。
- 为面向现代日常表达，本轮**排除了古文占主的维基文库**。最终保留 FineWeb2 3875 篇、维基百科 229 篇，共 4104 篇；训练 3713 篇、验证 193 篇、测试 198 篇。详细拒绝原因见 `prepared-curated-expanded/manifest.json`。
- 继续使用旧模型的同一分词器，避免模型词嵌入与新词表不匹配。新训练集约 619 万 token，验证集约 31 万 token，测试集约 25 万 token。

这只是启发式筛选，并未完成网页内容逐条审查、完整的近似去重、年龄适宜性验证或版权逐项核验。FineWeb2 取的是接口前 6000 行，不是有代表性的随机样本；其中仍可见旧网页、商业材料和口语表达。扩充后的训练集比旧版大，但依然远不足以保证流畅对话。

## 训练

新阶段使用 `configs/pet-curated.json`：约 3148 万参数、BF16、学习率 `1e-4`、2000 步。从 `data/petlm/models/three-source-extension/latest.pt` 加载权重；在新数据上重置优化器和阶段步数，输出到 `data/petlm/models/curated-expanded/`。原模型检查点保持不变。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run train --config configs/pet-curated.json --tokens data/petlm/tokens-curated-expanded --output data/petlm/models/curated-expanded --adapt-from data/petlm/models/three-source-extension/latest.pt --max-seconds 600
```

已完成全部 2000 步，新增处理约 819 万 token，耗时约 217 秒。阶段内验证损失从第 200 步的 6.458 降至第 2000 步的 5.944。

若将来从**未完成**的这个阶段恢复，可使用 `--resume`，配置、数据与总步数必须保持一致：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run train --config configs/pet-curated.json --tokens data/petlm/tokens-curated-expanded --output data/petlm/models/curated-expanded --resume data/petlm/models/curated-expanded/latest.pt --max-seconds 600
```

当前已达 2000／2000 步，重复运行不会增加训练量。继续扩充数据时，应新建快照和输出目录，按新实验重新评估。

## 结果评估

对**同一份新验证集、相同抽样窗口**各评估 50 批、25600 token：

| 模型 | 验证损失 | 困惑度 |
| --- | ---: | ---: |
| 扩充前 1500 步检查点 | 6.602 | 736.7 |
| 扩充后 2000 步检查点 | 5.948 | 383.0 |

新模型在留出的测试集上的损失为 5.949、困惑度约 383.3。两模型用同一分词器和新验证集比较；不能将旧报告中的验证损失与新报告直接比较，因为数据集不同。测试集本轮只做最终评估，未用于更新模型。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.evaluate --checkpoint data/petlm/models/three-source-extension/latest.pt --tokens data/petlm/tokens-curated-expanded --batches 50
.\.venv\Scripts\python.exe -X utf8 -m petlm.evaluate --checkpoint data/petlm/models/curated-expanded/latest.pt --tokens data/petlm/tokens-curated-expanded --batches 50
.\.venv\Scripts\python.exe -X utf8 -m petlm.run generate --tokens data/petlm/tokens-curated-expanded --checkpoint data/petlm/models/curated-expanded/latest.pt --prompt "今天我们一起学习科学。" --new-tokens 80 --seed 42
```

文本预测有改善，但人工检查的续写仍有重复、断句和不连贯之处。下一阶段需要更大且更干净的现代中文语料，以及单独的宠物对话训练与多轮对话评估。不要把目前模型用于可靠学习辅导。
