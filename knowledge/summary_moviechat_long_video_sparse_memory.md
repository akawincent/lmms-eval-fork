# MovieChat 论文阅读笔记（arXiv:2307.16449）

- 论文标题：MovieChat: From Dense Token to Sparse Memory for Long Video Understanding
- arXiv：2307.16449
- 读取方式：基于 arXiv `src` LaTeX 源码（非 PDF）
- 阅读时间：2026-03-10

## 1. 核心问题

论文要解决的是长视频理解（>10K 帧）中的三大瓶颈：

1. 计算复杂度高（帧太多，逐帧编码/注意力成本高）。
2. 显存与内存成本高（无法长期保留密集 token）。
3. 长时程时序关联弱（容易丢失远距上下文）。

核心思想是把“密集 token”压缩成“稀疏记忆”，用短期记忆 + 长期记忆机制在效率和信息保真之间做平衡。

## 2. 方法概览

MovieChat 的主链路：

1. 帧级视觉特征提取：用图像模型（EVA-CLIP ViT-G/14 + BLIP-2 Q-former）做 sliding window 编码。
2. 短期记忆（Short-term）：固定长度 FIFO 缓冲，保留最近密集帧 token。
3. 记忆巩固（Memory Consolidation）：周期性计算相邻帧相似度，贪心合并最相似邻接帧，直到压缩到 `R_L`。
4. 长期记忆（Long-term）：存储压缩后的稀疏表示，降低总 token 量并保留长程信息。
5. 推理：
   - Global mode：用长期记忆回答整段视频问题。
   - Breakpoint mode：拼接长期记忆 + 短期记忆 + 当前时刻特征，回答局部时刻问题。

文中超参（附录）给出的默认配置：

- sliding window size = 16 frames
- short-term memory = 18 frames × 32 tokens/frame
- long-term memory = 256 frames
- consolidation length = 2

## 3. 数据集：MovieChat-1K

论文提出 MovieChat-1K：

- 1K 长视频（多来自电影/剧集片段）
- 约 14K 人工标注（正文写 13K，摘要/其他位置写 14K，论文内表述略有不一致）
- 每视频包含：
  - 1 条 dense caption
  - Global QA（3 对）
  - Breakpoint QA（10 对，带时间戳）

它强调长视频细粒度理解，不是只做稀疏采样后的短视频问答。

## 4. 关键实验结果（论文中的主表）

### 4.1 短视频 QA（GPT-3.5 辅助评测）

MovieChat 在 MSVD-QA / MSRVTT-QA / ActivityNet-QA 上分别达到：

- Acc：75.2 / 52.7 / 45.7
- Score：3.8 / 2.6 / 3.4

### 4.2 MovieChat-1K 长视频 QA（三方平均：GPT-3.5 + Claude + 人评）

- Global：62.3 Acc / 3.23 Score
- Breakpoint：48.3 Acc / 2.57 Score

基线（Video Chat / Video-LLaMA / Video-ChatGPT）在帧数上分别约 32/32/100，而 MovieChat 可处理约 2048 帧输入，且性能更优。

### 4.3 长视频生成质量（Global）

MovieChat 在 CI/DO/CU/TU/CO 五个维度上分别是：

- 3.11 / 2.93 / 3.24 / 3.17 / 3.25

均高于对比基线。

### 4.4 消融

去掉记忆机制（w/o MM）会明显退化：

- QA（Global）：67.8 → 51.4（Acc）
- QA（Breakpoint）：50.4 → 38.2（Acc）
- 生成质量多个维度同样下降

说明“短期 + 长期 + 合并压缩”不是装饰模块，而是性能关键。

## 5. 局限性（论文自述）

1. 感知能力受限于预训练短视频模型上限。
2. 时间处理不够精细，只能粗粒度估计事件时长比例。

## 6. 与当前 lmms-eval-fork 的对照

仓库里已存在完整 MovieChat 相关实现：

- 任务：`lmms_eval/tasks/moviechat/`
- 模型：`lmms_eval/models/simple/moviechat.py`
- 变体：`lmms_eval/models/simple/llava_onevision_moviechat.py`

### 6.1 一致点

1. 任务拆分成 `moviechat_global` 和 `moviechat_breakpoint`，与论文双推理模式一致。
2. 模型侧实现了短期缓冲、相邻相似度合并、长期缓冲，整体思想与论文算法对齐。

### 6.2 可注意的偏差/风险

1. 超参与论文默认值并不完全一致：
   - `moviechat.py` 默认 `sliding_window_length=8`（论文附录是 16）。
   - `llava_onevision_moviechat.py` 默认 `long_memory_length=64`（论文配置是 256）。
2. 评测协议和论文描述有差异：
   - 论文强调 GPT-3.5/Claude/人评及人工一致性过滤；
   - 当前 `tasks/moviechat/utils.py` 默认评测模型环境变量回落到 `gpt-4o-2024-11-20`，且未实现论文里“人工过滤不一致判分”的流程。
3. `moviechat` 任务仍走 `doc_to_text + doc_to_visual` 旧范式，若后续新增变体任务，建议优先转向 `doc_to_messages`。

## 7. 对本仓库可落地的改进建议

1. 增加“论文复现实验配置档”（例如 `moviechat_paper_defaults`），统一 `sliding_window_length=16`、`long_memory_length=256` 等关键值，便于结果可比。
2. 将 MovieChat 评测脚本协议化：显式支持 `judge_model`、多评审融合、可选一致性过滤（复现论文评测链路）。
3. 给 breakpoint 模式补充单测，确认时间戳切片/上下文构建与任务定义一致。
4. 若后续扩展新长视频任务，优先用 `doc_to_messages` 新接口，保持与仓库当前任务开发规范一致。

