# 自建中文语言模型：数据接入与训练

最新扩充与训练结果见 [EXPANSION.md](EXPANSION.md)。本页较早的三源命令用于说明基础流程；最新检查点在 `data/petlm/models/curated-expanded/latest.pt`。

本项目使用 PyTorch，从随机权重训练模型，不加载任何现成模型权重。默认配置为 8 层、512 隐藏维度、8 个注意力头、256 token 上下文、12000 词表，约 3148 万参数。定位是训练研究原型，不是已经具备可靠聊天能力的通用大模型。

## “直接对接”的含义

可以对接官方数据接口或批量转储，由程序读取文本并训练，无需手工复制。不是把一个网址交给模型，它就自动获得该站点知识。另一种方式是问答时检索资料后把相关内容提供给模型；本项目当前实现的是训练数据接入，不是检索增强问答。

当前支持：

| 来源 | 方式 | 当前状态 |
| --- | --- | --- |
| 中文维基百科 | 官方日期转储，流式解压 XML 并提取正文 | 已取得 300 篇样本，导入后保留 292 篇 |
| 中文维基文库 | 官方日期转储，流式解压 XML 并提取正文 | 已取得 300 篇样本，导入后保留 297 篇 |
| FineWeb2 | 官方 datasets-server 的 `cmn_Hani` 中文配置分批读取；也支持本地 JSONL 导入 | 已下载 1000 篇完整样本，清洗去重后保留 990 篇，已参与三源短训练 |
| CLUECorpus2020 | 经授权取得文件后本地导入 | 官方 README 要求申请，未取得语料；不是已经接入 |

官方来源：

- <https://dumps.wikimedia.org/zhwiki/>
- <https://dumps.wikimedia.org/zhwikisource/>
- <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2>
- <https://github.com/CLUEbenchmark/CLUECorpus2020>

CLUE 官方要求向 `CLUEbenchmark@163.com` 申请，说明研究目的、计划、机构和申请者信息，并承诺不向第三方提供语料。程序不会代发申请，不会以 CLUECorpusSmall 冒充 CLUECorpus2020。

## 已完成的小规模验证

当前三源快照包含 1579 篇：FineWeb2 990 篇、维基百科 292 篇、维基文库 297 篇。划分为训练 1435 篇、验证 73 篇、测试 71 篇；已完成分词、1000 步基础训练，并从该检查点继续训练至 1500 步。最新检查点保存于 `data/petlm/models/three-source-extension/latest.pt`，原 1000 步检查点仍在 `data/petlm/models/three-source/latest.pt`。这仍是小规模实验，不是成熟模型。样本不是随机代表性样本，仅选取各来源开头的符合条件记录。文库包含古文等内容，正式宠物训练前还需筛选。

`data/petlm/download-status.json` 记录各源访问状态，单独获取一个来源不会覆盖其他来源状态。`prepared-three-source/manifest.json` 明确列出仍缺失的 CLUE；原来的 `prepared/` 两源快照保留不变。`synthetic-verification/` 下的内容是单独的人工测试夹具，不属于正式语料，也没有混入三源样本。

FineWeb2 样本保存在 `data/petlm/raw/fineweb2.jsonl`，旁边的 `fineweb2.metadata.json` 记录配置、抓取时间、官方数据卡许可与本地文件 SHA256。官方配置是 `cmn_Hani`，包含汉字中文，不代表全部都是简体。数据集查看接口没有固定到指定提交，因此提交号只记录获取时观察值，文件哈希标识实际样本。官方数据集卡声明 ODC-BY；原始网页内容的权利仍需按具体内容核实。

下载器对暂时性连接错误和限流最多尝试三次，拒绝截断的正文、语言不符记录及重复 ID。它用于小样本验证，不能替代全量 Parquet 语料下载。

## 获取和导入

以下命令在项目目录的 PowerShell 中运行。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.fetch --source all --limit 300
.\.venv\Scripts\python.exe -X utf8 -m petlm.fetch --source fineweb2 --limit 1000
```

单个维基源最多读取 64 MiB 压缩数据；FineWeb2 按行读取。已完成文件不会覆盖或自动扩容，重复执行会复用现有样本。请求的来源失败或待申请时，命令会返回非零状态并保存原因；仅请求 FineWeb2 不会因为 CLUE 待申请而报失败。

下载并不自动等于纳入训练。导入时必须记录真实来源和适用条款；条款字段是出处说明，不是程序自动授予许可。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.data import --source fineweb2 --path data/petlm/raw/fineweb2.jsonl --origin https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 --license-note "ODC-BY; original website content rights may vary" --limit 10000
```

CLUE 经申请后，用相同命令，将 source 改为 `clue2020`，path 改为获准使用的本地文件夹，origin 改为官方项目地址，并填入申请授权范围。支持 TXT（空行分文档）、JSONL（每行有 text 字段）、维基 XML，以及对应 gzip/bzip2 压缩文件；暂不支持直接导入 Parquet。

## 数据快照与分词

数据存入 SQLite，保留来源和文档 ID，进行 NFKC 标准化、手机号/邮箱基础替换、中文比例和长度筛选。按空白归一后的正文哈希进行完全去重，并稳定分为约 90%/5%/5% 三组，分词器只用训练集拟合。

这不是完整的内容审查：近似重复、其他隐私字段、广告、低质量和不适龄内容仍需进一步筛选。简繁体均保留。维基文库版权需按作品核实，正式使用时保留署名等所需信息。

补齐四源后导出新快照，避免破坏已有实验：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.data export --output data/petlm/prepared-four-source
.\.venv\Scripts\python.exe -X utf8 -m petlm.run tokenize --data data/petlm/prepared-four-source --tokens data/petlm/tokens-four-source --vocab-size 12000
```

默认导出要求四种来源都有独立文档。只有明确使用 `--allow-partial` 才允许部分来源，清单仍保留缺失项。当前三源试验使用该选项，不代表四源完成。新数据或新分词器应建立新的实验，不能载入旧快照冒充相同数据续训。

## 训练与继续训练

三源数据的初始 1000 步计划已完成，下面的命令是历史记录；同一输出目录再次启动从头训练会被阻止，以免覆盖模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run train --tokens data/petlm/tokens-three-source --output data/petlm/models/three-source --max-seconds 600
```

从已完成计划继续，使用 `--extend-from`、更大的总步数以及新的输出目录。已经运行并完成的 1000→1500 步命令如下，扩展阶段使用较低学习率，保留原检查点：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run train --tokens data/petlm/tokens-three-source --output data/petlm/models/three-source-extension --extend-from data/petlm/models/three-source/latest.pt --steps 1500 --learning-rate 0.00003 --max-seconds 600
```

第 1000 步验证误差为 6.3663，第 1500 步为 6.3178，改善幅度约 0.0485。新增处理 204.8 万 token。较低的验证误差不代表具备流畅聊天能力。下阶段应优先扩充和筛选语料、准备对话数据，而不是无限重复这批小样本。

以新四源快照开始训练：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run train --tokens data/petlm/tokens-four-source --output data/petlm/models/four-source --max-seconds 600
```

默认最多训练约 10 分钟，每完成一整步检查时间，保存后退出；默认总计划为 1000 步。使用 BF16（硬件不支持时退回 FP16）、梯度累积、梯度裁剪和 AdamW。验证集只评估，不更新参数，测试集保持留出。训练窗口按 token 随机抽取，当前不提供按来源加权采样，长文档会贡献更多 token。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run train --tokens data/petlm/tokens-four-source --output data/petlm/models/four-source --resume data/petlm/models/four-source/latest.pt --max-seconds 600
```

同一阶段的 `--resume` 要求模型配置、总步数、分词器和数据保持一致。新的扩展阶段 `--extend-from` 要求模型结构、分词器和数据一致，允许更大的总步数及较低学习率，并保留模型、优化器和随机状态。扩展阶段中断时，对新目录使用 `--resume`，保持该阶段的 `--steps` 和 `--learning-rate` 不变。遇到异常使用最近一次成功保存的检查点；不是每一步都保存。`--max-seconds 0` 取消时间限制，只适合明确打算长时间训练时使用。

## 检查生成与曲线

当前三源扩展模型的生成命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m petlm.run generate --tokens data/petlm/tokens-three-source --checkpoint data/petlm/models/three-source-extension/latest.pt --prompt "我们一起学习科学"
.\.venv\Scripts\python.exe -m tensorboard.main --logdir data/petlm/models --host 127.0.0.1 --port 6006
```

生成是续写，不是聊天接口。短训练产生重复、乱码或不可靠内容是预期现象，不能作为学习辅导答案。训练结果见各模型目录的 report.json；训练结束并不代表质量合格。

## 验证代码

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 tests/smoke_training.py
```

测试覆盖因果遮罩、跨源完全去重、缺失来源阻断、维基命名空间与重定向过滤、基础隐私过滤、FineWeb2 配置与分页、截断文本拒绝、网络重试和独立来源状态保存，以及分词、GPU 训练、中途保存、继续更新参数和生成流程。人工测试夹具的评估值不衡量真实语言能力。
