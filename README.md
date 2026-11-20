# 电子宠物 Machine Pet

项目源码、训练与仿真工具已收录。可公开数据、使用范围以及未上传的本地资产见 [发布说明](DATA_RELEASE.md)。

已增加从零预训练程序，数据接入、当前完成情况和操作命令见 [TRAINING.md](TRAINING.md)。四源完整训练尚未完成，具体来源状态以数据清单为准。

最新语料扩充、筛选、继续训练和评估结果见 [EXPANSION.md](EXPANSION.md)。

最新一轮语义改进实验、评估和 VS Code 本地对话入口见 [SEMANTICS.md](SEMANTICS.md)；早期对话实验见 [DIALOGUE.md](DIALOGUE.md)。

2026-11-03 从随机权重重新训练的中文 v5 模型、数据规模、实测指标和当前限制见 [CHINESE_V5.md](CHINESE_V5.md)。

2万条多轮对话候选的逐条审核流程和1000题隔离盲测集见 [DIALOGUE_EXPANSION.md](DIALOGUE_EXPANSION.md)。候选数据默认不进入训练，只有明确人工通过的条目才可导入。

最新高质量中文语料 v7、基础预训练和对话重训结果见 [PRETRAIN_V7.md](PRETRAIN_V7.md)；上一版记录见 [PRETRAIN_V6.md](PRETRAIN_V6.md)。

继续训练与多种对话采样实验见 [MODEL_IMPROVEMENT_V8.md](MODEL_IMPROVEMENT_V8.md)。基础模型指标继续改善，但对话回归退化，因此默认聊天模型仍保留 v7。

真人自然中文对话的质量重筛、审核口径和独立参考集见 [NATURAL_DIALOGUE_REVIEW.md](NATURAL_DIALOGUE_REVIEW.md)。当前数据是用户整批确认，不是逐条人工审核。

自然对话替换训练与新28题评估见 [MODEL_IMPROVEMENT_V9.md](MODEL_IMPROVEMENT_V9.md)。v9在两套独立评估上都退化，因此默认聊天模型仍保留 v7。

最新的逐条程序审核、参考答案修正和低采样对比见 [MODEL_IMPROVEMENT_V10.md](MODEL_IMPROVEMENT_V10.md)。v10.1已修复v9的部分退化，但独立指标仍未超过v7，所以默认模型不变。

本目录是 Windows + VS Code 的 Python 项目，采用本地从零训练模型的路线。当前有实验性文字对话入口，但回复仍不可靠；语音和长期记忆尚未实现。环境检查不下载预训练权重，不调用云端模型，也不需要 API 密钥。

## GPU 训练环境

- Python 3.11，使用项目 `.venv`。
- PyTorch 2.11.0 + CUDA 12.8，来自 PyTorch 官方 CUDA 软件源。
- NumPy：数值处理；SentencePiece：训练自己的分词器；TensorBoard：查看训练曲线。
- 使用 Windows 原生 NVIDIA GPU，不需要另外安装完整 CUDA Toolkit 或 WSL。

`nvidia-smi` 中的 CUDA 版本表示驱动支持能力，不要求与 PyTorch 自带运行库的版本完全一致。

### 运行训练检查

在 VS Code 的“运行和调试”下拉框选择“电子宠物：验证 GPU 训练”，按 F5。或在项目终端执行：

```powershell
.\.venv\Scripts\python.exe check_gpu.py
```

它用随机初始化的小型 Transformer 和模拟数据验证 GPU 前向计算、混合精度、反向传播、AdamW 参数更新、模型与优化器保存及恢复训练。数据完全在本地生成。这不是宠物模型的正式训练，也不是吞吐量基准测试。

结果保存在 `data/gpu-check/report.json`，测试权重保存在 `data/gpu-check/synthetic-checkpoint.pt`，均被 Git 忽略。

在“终端 → 运行任务”中选择“电子宠物：查看训练曲线”，然后访问 <http://127.0.0.1:6006>。关闭该任务或按 Ctrl+C 停止服务。

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir data/gpu-check/runs --host 127.0.0.1 --port 6006
```

笔记本正式长时间训练时接通电源、保持通风。其他使用显卡的应用会占用训练可用显存，可按需自行关闭。检查程序只做短时间训练。

## 在 VS Code 中运行

1. 用 VS Code 打开整个“机器”文件夹。
2. 按 `Ctrl+Shift+P`，选择 `Python: Select Interpreter`，选择项目内 `.venv\Scripts\python.exe`。如果没有出现在列表里，选择“输入解释器路径”并定位该文件。
3. 打开 `main.py`，按 `F5` 运行调试，或按 `Ctrl+F5` 直接运行。也可按 `Ctrl+Shift+B` 执行环境检查任务。

为避免 Windows 的脚本执行策略阻止环境激活，工作区关闭了自动激活。运行和调试配置已指定项目解释器；在终端中请明确使用下面的路径，不要直接依赖全局 `python` 或 `pip`：

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 本地模型路线

旧的 `.env.example` 是早期 API 方案的空模板，本地训练不使用其中的字段，无需填写。用户记忆保存在独立数据库中，不通过每次聊天后更新模型权重来保存。

## 开发顺序

1. 收集有权用于训练的中文文本，去重并划分训练集、验证集和测试集。
2. 训练分词器，从随机权重开始训练小型语言模型，并测量速度与显存占用。
3. 用审核过的宠物对话训练角色与记忆使用能力。
4. 接入 SQLite 用户档案、记忆、桌面交互和独立提醒程序。
5. 后续再扩展语音和实体控制。

`httpx` 和 `python-dotenv` 是已有基础依赖，可用于后续网络请求和配置读取。SQLite 随 Python 提供。语音依赖在选定方案后再安装。

## 文件说明

- `main.py`：本地环境检查入口。
- `check_gpu.py`：真实 GPU 训练流程检查，不含预训练模型。
- `requirements.txt`：项目基础依赖。
- `requirements-gpu.txt`：GPU 版 PyTorch 与官方软件源。
- `requirements-training.txt`：分词、数值处理和训练曲线工具。
- `requirements-lock.txt`：含训练工具的环境版本快照。
- `.vscode/`：解释器、运行、调试和依赖安装任务配置。
- `.env.example`：无密钥的模型配置示例。

如需在支持相同 CUDA 版本的 Windows NVIDIA 电脑重建环境，先用 Python 3.11 创建 `.venv`，安装 GPU 组件，再安装快照：

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```
