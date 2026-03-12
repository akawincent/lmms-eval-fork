# Cambrian-S 论文与源码阅读总结

- 论文: https://arxiv.org/pdf/2511.04670
- 代码: https://github.com/cambrian-mllm/cambrian-s
- 阅读方式:
  - `read-arxiv-paper`: 下载并解压 TeX 源码 (`https://arxiv.org/src/2511.04670`)，主文件为 `release.tex`。
  - `read-github`: 先尝试 `gitmcp.io` 路径解析；当前环境缺少 `scripts/gitmcp.py` 且 `gitmcp` 直连仅返回前端页，因此回退到仓库源码快照（`codeload.github.com`）做代码级阅读。

## 1. 这篇论文在讲什么

论文核心观点是: 现有视频 MLLM 大多仍是“反应式 + 长上下文堆料”范式，离真正的空间智能还远。作者提出“spatial supersensing”层级，把能力分成:

1. 语义感知（看见并命名）
2. 流式事件认知（持续观察和记忆）
3. 隐式 3D 空间认知（理解空间关系和变化）
4. 预测式世界建模（基于预测误差组织注意力、记忆与学习）

论文结论不是“我们已解决 supersensing”，而是:

- 先用新 benchmark 证明现范式不足；
- 再用数据/训练 scaling（Cambrian-S）把旧范式推到很强；
- 最后给出 predictive sensing 原型，展示“预测误差驱动记忆/分段”确实能显著更好地处理超长视频流。

## 2. Benchmark 与关键实证

### 2.1 VSI-Super 的设计

论文提出 `VSI-Super` 两个任务:

- `VSO` (Recall): 在超长视频里顺序回忆“异常物体”出现位置（10/30/60/120/240 分钟）。
- `VSC` (Count): 在多场景连续视频里累积计数（10/30/60/120 分钟，含 streaming 评测）。

设计目标是“抗暴力长上下文”:

- 视频时长可任意扩展；
- 要求选择性记忆与结构化更新，不是把所有帧都塞进上下文。

### 2.2 现有强模型在新任务上的失效

论文中 Gemini-2.5-Flash 在常规 benchmark 很强，但在 VSI-Super 明显掉队:

- VideoMME: 81.5
- VideoMMMU: 79.2
- VSI-Bench: 45.7
- VSO@60min: 41.5
- VSC@60min: 10.9
- 120min 已出现 Out-of-Context

并且 VSC 预测值随真实计数不增长，出现“饱和到小常数”的模式。

## 3. 数据 scaling（Cambrian-S）做到了什么

论文第二部分验证“是不是数据问题”。做法:

- 构建 `VSI-590K`（空间感知指令数据）；
- 在 Cambrian-1 基础上做 4-stage 训练，得到 Cambrian-S（0.5B/1.5B/3B/7B）。

关键结果:

- `Cambrian-S-7B` 在 VSI-Bench 达到 67.5，超过 Gemini-2.5 Pro 的 51.5（+16 绝对点）。
- 在 debiased 版本上仍保持优势（59.9），说明不是纯语言捷径。
- 但到了 VSI-Super 仍明显失效，尤其长时持续感知场景。

这支持论文主张: scaling 很重要，但不够。

## 4. Predictive Sensing 原型方法

### 4.1 训练侧

在 stage-4 上增加 LFP（Latent Frame Prediction）头:

- 一个两层 MLP，预测“下一帧 latent”；
- 联合优化:
  - 主任务语言损失
  - `MSE` + `cosine` 的 LFP 辅助损失
- 论文中还说明使用专门 1 FPS 视频子集进行该目标学习。

### 4.2 推理侧

用预测误差作为 surprise 信号:

- `surprise = 1 - cosine(pred_next_latent, actual_next_latent)`
- 用 surprise 做两件事:
  - 记忆管理（VSO）: 低惊讶帧压缩/淘汰，高惊讶帧保留；
  - 事件分段（VSC）: 遇到 surprise 峰值触发段落结算和累计。

论文报告该方法在 VSI-Super 上相对长上下文/实时商业模型更稳，且在 streaming 评测下能维持更高 MRA。

## 5. 源码对照: 论文方法在仓库里如何实现

### 5.1 训练脚本层

- `cambrian/scripts/cambrians_7b_s4.sh`
  - 标准 stage-4 空间视频微调。
- `cambrian/scripts/cambrians_7b_lfp_s4.sh`
  - 在 s4 基础上增加:
    - `--nfp_head True`
    - `--nfp_mse_loss_weight 0.1`
    - `--nfp_cosine_loss_weight 0.1`

### 5.2 模型层（LFP head 与损失）

- `cambrian/model/cambrian_arch.py`
  - 根据配置创建 `nfp_head`（Linear + GELU + Linear）。
  - 将 `nfp_head` 与损失权重注入 config。
- `cambrian/model/language_model/cambrian_qwen2.py`
  - `nfp_loss()` 同时计算 MSE 与 cosine。
  - 训练时把 `nfp_loss` 与语言建模损失相加返回。

### 5.3 数据管线层（NFP 样本）

- `cambrian/train/train_fsdp.py`
  - `ModelArguments` 包含 `nfp_head` 与权重参数。
  - 数据中可用 `dat["nfp"] = True` 标记 NFP 视频样本。
  - 构造 `nfp_token_indices` 与 `nfp_loss_masks`，最后进入 collator，供模型计算 NFP 目标。

### 5.4 评测实现层（VSR/VSC/Streaming）

- `lmms-eval/lmms_eval/models/simple/cambrians_vsr.py`
  - 每帧计算 surprise 分数；
  - 低 surprise 可做 KV downsample；
  - 超预算时执行 `drop`/`drop_merge` consolidation；
  - 可选 `retrieval_topk` 仅检索关键视觉块。
- `lmms-eval/lmms_eval/models/simple/cambrians_vsc.py`
  - surprise 超阈值触发事件分段并生成分段计数。
- `lmms-eval/lmms_eval/models/simple/cambrians_vsc_streaming.py`
  - 在分段逻辑上加入 `query_times` 流式查询，输出多时刻累计结果。
- `lmms-eval/qwen2_monkey_patch.py`
  - 对注意力前向做 retrieval patch，支持按 cache block 相似度 top-k 取视觉块。

### 5.5 任务与指标

- `lmms-eval/lmms_eval/tasks/cambrians_vsr/*.yaml`
  - 数据集: `nyu-visionx/VSI-SUPER-Recall`
  - 指标: exact match accuracy
- `lmms-eval/lmms_eval/tasks/cambrians_vsc*.yaml`
  - 数据集: `nyu-visionx/VSI-SUPER-Count`
  - 指标: Mean Relative Accuracy (MRA)

## 6. 对你当前 lmms-eval fork 的直接启发

从工程角度，这篇工作最值得借鉴的不只是“模型更大/数据更多”，而是评测范式:

1. 明确区分“短视频 QA 能力”与“流式持续感知能力”。
2. 引入可无限扩展时长的 stress benchmark，避免固定窗口掩盖问题。
3. 把“预测误差驱动的记忆预算管理”作为可独立 ablate 的系统模块（阈值、压缩率、预算、检索 top-k）。
4. 在指标上加入随时长变化曲线，而不只看单点平均分。

如果你后续希望把这套思想迁移到你当前仓库（`lmms-eval-fork` 主干）而不是 `cambrian-s` 子仓库，我建议先从 task 侧开始: 先复刻 VSR/VSC 的数据处理和 metric 协议，再逐步接入 surprise-aware 模型 wrapper。
