# Thinking in Space (arXiv:2412.14171) 阅读总结

## 论文信息
- 标题: **Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces**
- 任务核心: 测试 MLLM 是否能基于视频构建空间表征并进行空间推理，而不只是做一般视频理解。
- 源码入口:
  - `~/.cache/nanochat/knowledge/2412.14171/camera_ready.tex`
  - `~/.cache/nanochat/knowledge/2412.14171/sec/suppl_refer.tex`

## 这篇论文做了什么
- 构建了一个视频空间智能评测集 **VSI-Bench**:
  - 288 个真实室内视频
  - 5000+ QA
  - 8 类任务（object count, abs/rel distance, rel direction, room/object size, route plan, appearance order）
- 定义了“视觉空间智能”分析框架，拆成 4 个能力轴:
  - 视觉感知
  - 语言智能
  - 时序处理
  - 空间推理（含关系推理 + 自我中心/世界中心坐标转换）
- 对 15 个视频 MLLM 做零样本评测，并做人类对照。
- 做了“模型如何思考空间”的两条 probing:
  - 语言侧：self-explanation + 错误类型分析
  - 视觉侧：让模型显式输出 cognitive map（10x10 网格中的物体中心）

## 关键实验设计
- 指标:
  - MCA 任务用 Accuracy。
  - 数值任务提出 **MRA (Mean Relative Accuracy)**，通过多个相对误差阈值平均，避免数值题只看 exact match。
- 基线:
  - random chance
  - frequency chance（永远选高频答案）
- 人类评测:
  - `VSI-Bench_tiny` 400 题用于人工标注与误差分析。

## 主要结论（高价值）
- MLLM 在该任务上“有能力但离人类差距大”:
  - 人类约 79%（tiny）
  - 最强模型明显低于人类（正文结论约低 33%）
- **瓶颈不是看不见，而是空间推理不足**:
  - 错误分析中约 71% 错误来自空间推理（关系推理 + 视角转换）
- 常见语言推理提示在空间任务上反而伤害性能:
  - Zero-shot CoT / Self-Consistency / ToT 在 VSI-Bench 平均不增反降
- 显式“认知地图”有帮助:
  - 相对距离任务中，用 cognitive map 提示可提升（示例 46 -> 56）
  - 用 GT cognitive map 还能更高，说明“地图质量”是关键限制
- 模型的空间表征呈现 **local 强、global 弱**:
  - 邻近目标距离判断准确率高，远距离关系显著退化

## 附录里值得注意的细节
- 数据构建质量控制比较严:
  - 统一三套 3D 数据集元信息
  - 多轮人审纠错与规则回流
- 评测协议复现性强:
  - 统一 prompt 模板
  - 默认 greedy
  - 公开不同模型帧数设置
- 一个有意思现象:
  - 输入重复视频两次（`[Video][Context][Video]`）有增益（48.8 -> 50.9）
- 文中“video-first / question-first”一句描述与表格数值有出入，建议以表格为准并自行复验。

## 对当前 lmms-eval 仓库的启发（可直接落地）
- 1) 新增空间智能任务组
  - 在 `lmms_eval/tasks/` 下按 8 子任务组织，复用现有 video task 模板。
  - 数值题实现 MRA 聚合函数，避免仅靠 exact/fuzzy match。
- 2) 增加“认知地图”评测子流程
  - 单独任务要求模型先输出 map JSON，再答题（两阶段或单回合结构化输出）。
  - 评估 map 的局部/全局距离一致性，作为中间诊断指标。
- 3) 把“提示词策略 ablation”作为标准实验
  - baseline vs CoT vs self-consistency vs ToT
  - 明确区分“语言推理增益”与“空间推理增益”
- 4) 输入编排实验纳入 workflow
  - 比较 video-first / question-first
  - 比较单次视频输入 vs 重复输入
  - 与模型上下文长度、帧采样策略联动

## 我建议的最小复现实验包（在 lmms-eval 中）
- 一个 tiny 子集（几百题）用于快速回归。
- 一个 full 子集用于正式对比。
- 统一输出三类结果:
  - 任务主分数（ACC/MRA）
  - 错误类型分布（至少半自动）
  - cognitive map 质量分（local/global）

## 一句话总结
这篇论文最重要的贡献不是“又一个视频 benchmark”，而是把“空间智能失败原因”定位到可操作层面：**MLLM 已具备一定局部空间表征，但全局一致空间模型和视角变换能力仍是主瓶颈；单纯加语言链式推理往往无效，显式空间表征（认知地图）更有前景。**
