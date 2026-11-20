# v6r2 训练记录

日期：2026-11-08

## 数据隔离

- 训练只新增 v6r2 开发集 160 条，每项能力 40 条。
- 内部测试集 160 条仅用于验证和回归。
- 冻结集 160 条未用于训练、检查点选择、生成或评分。
- 冻结提示与训练快照重叠为 0。

## LoRA 训练

四个适配器均从同一个冻结基础检查点训练，rank 8，目标 160 步，实际均完成 160 步。

| 能力 | 未适配验证损失 | 最佳验证损失 | 训练参数量 |
|---|---:|---:|---:|
| 安全行为 | 3.9103 | 0.6735 | 2,097,152 |
| 记忆管理 | 3.7897 | 0.6478 | 2,097,152 |
| 移动澄清 | 4.3963 | 0.6402 | 2,097,152 |
| 格式控制 | 3.4670 | 0.7520 | 2,097,152 |

基础模型约 2.27 亿参数保持冻结。

## 开发与内部回归

- 能力路由：开发集 160/160，内部测试集 160/160。
- 严格最低要素通过：纯模型开发集 0/160，内部测试集 0/160。
- 严格最低要素通过：保护应用开发集 0/160，内部测试集 0/160。
- 纯模型内部测试要素覆盖率：安全 20%，记忆 28%，移动 29%，格式 19%。

损失下降说明模型学习了训练分布，但生成仍常遗漏对象、处理边界、完整格式或必要追问。保护状态机也尚未覆盖 v6r2 的新约束，因此不能用训练损失替代生成验收。

## 决定

- 保留四个检查点作为实验候选，不切换当前应用默认配置。
- 不读取或评分 v6r2 冻结集；先在新的开发循环中改善生成与保护状态机。
- 真实电机、车轮、GPIO/串口动作和自主导航继续禁用。

## 证据

- `data/petlm/dialogue/prepared-human-review-v6r2-capabilities-v1/manifest.json`
- `data/petlm/models/dialogue-v6r2-lora-human-safety-r8/report.json`
- `data/petlm/models/dialogue-v6r2-lora-human-memory-r8/report.json`
- `data/petlm/models/dialogue-v6r2-lora-human-motion-r8/report.json`
- `data/petlm/models/dialogue-v6r2-lora-human-format-r8/report.json`
- `data/petlm/evaluation/human-review-v6r2-new-adapters-regression-v2/report.json`
