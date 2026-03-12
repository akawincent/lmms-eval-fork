# TALL / CTRL / Charades-STA 论文与 `lmms_eval` 对照总结

论文：`TALL: Temporal Activity Localization via Language Query`  
arXiv：`1705.02101`  
源码入口：`~/.cache/nanochat/knowledge/1705.02101/egpaper_final.tex`

## 一句话总结

这篇论文提出了一个新任务 `TALL`（Temporal Activity Localization via Language）：给定一段未裁剪视频和一句自然语言描述，输出该事件在视频中的开始和结束时间。为了解这个任务，作者提出了 `CTRL`（Cross-modal Temporal Regression Localizer），并在原始 `Charades` 数据集上半自动构造了新的 benchmark `Charades-STA`。  

对我们当前仓库来说，最重要的一点是：`lmms_eval/tasks/charades_sta` 并没有复现论文里的 CTRL 训练/检索/回归流程，而是把 `Charades-STA` 改写成了一个“给视频 + 给句子，让视频模型直接生成起止时间”的生成式 benchmark。

## 1. 论文核心问题

论文的出发点是：传统 temporal action localization 只能在预定义动作类别上做分类式定位，但真实用户更自然的输入其实是一句开放词表的语言描述，比如“person turn a light on”或者更复杂的动作短句。  

因此论文把问题定义为：

- 输入：一段长视频 `V` 和一句语言查询 `s`
- 输出：该查询对应事件的 `(start_time, end_time)`

这就是 `TALL`。

## 2. CTRL 方法在做什么

CTRL 的核心不是“直接从整段视频一次性输出答案”，而是“先滑窗产生候选 clip，再做语言-视频对齐，再回归边界”。

它大致分四块：

1. `Visual Encoder`
   - 对长视频做 temporal sliding windows，得到很多候选 clip。
   - 每个 clip 不只看自己，还看前后 context clip。
   - 用 `pre-context + central clip + post-context` 拼成视觉表示。

2. `Sentence Encoder`
   - 把语言句子编码成向量。
   - 论文里实验了 LSTM 和 Skip-thought。

3. `Multi-modal Processing`
   - 视觉向量和句子向量做逐元素乘、逐元素加、拼接后 FC。
   - 得到联合表示。

4. `Temporal Regression`
   - 对每个候选 clip 输出：
     - 一个 alignment score
     - 一个时间边界回归偏移

论文特别强调：

- 他们尝试了 parameterized regression 和 non-parameterized regression
- 在时间边界上，`non-parameterized offset` 更好
- 直觉是：图像框可以缩放归一化，但动作持续时间本身就是时间轴上的真实尺度，不像图像框那样天然适合参数化缩放

## 3. Charades-STA benchmark 是怎么来的

`Charades-STA` 不是原始 Charades 自带的标注，而是作者在 `Charades` 上追加出来的 sentence-temporal annotation。

论文里的构造流程是半自动三步：

1. `Sentence decomposition`
   - 把原来的视频级长描述拆成子句
   - 用人工收集的连接词规则切分
   - 再补上主语

2. `Keyword matching`
   - 把 Charades 的活动类别关键词和子句做匹配
   - 匹配上后继承对应动作标注的时间段

3. `Human check`
   - 作者人工检查子句是否通顺、时间段是否真的匹配

论文里给出的统计是：

- train: `13898` 个 clip-sentence pairs
- test: `4233` 个 clip-sentence pairs
- complex test queries: `1378`
- 非复杂句平均词数：`6.3`
- 复杂句平均词数：`12.4`

这说明 benchmark 的基本单元就是：

- 一个视频
- 一句描述某个局部事件的短句
- 一个对应的 `[start, end]`

也就是说，`Charades-STA` 本质上是 sentence-to-time-span grounding 数据集。

## 4. 论文里的评测方式

论文使用的是 temporal grounding 常见指标：

- `R@n, IoU=m`

含义是：

- 对一个 query，看 top-n 预测里是否至少有一个时间段与 GT 的 IoU 大于等于阈值 `m`
- 再对所有 query 取平均

在 `Charades-STA` 上，论文表 2 报告的最好结果是 `CTRL(reg-np)`：

- `R@1, IoU=0.5 = 23.63`
- `R@1, IoU=0.7 = 8.89`
- `R@5, IoU=0.5 = 58.92`
- `R@5, IoU=0.7 = 29.52`

## 5. Hugging Face 上这个 benchmark 的实际内容

当前仓库的 task 指向的是：

- dataset repo: `lmms-lab/charades_sta`

我实际查看了 Hugging Face 上的数据仓库，看到：

- 只有 `default/test` 一个 split
- `README.md` 里声明的 feature 是：
  - `video: string`
  - `caption: string`
  - `timestamp: sequence<float16>`
- `test` split 当前是 `3720` 条，不是论文里的 `4233`
- 仓库里还包含：
  - `data/test-00000-of-00001.parquet`
  - `Charades_v1_480_part_1.zip`
  - `Charades_v1_480_part_2.zip`
  - `Charades_v1_480_part_3.zip`
  - `Charades_v1_480_part_4.zip`

我从 parquet 中抽到的前 3 条样本是：

```python
{'video': '3MSZA.mp4', 'caption': 'person turn a light on.', 'timestamp': [24.296875, 30.40625]}
{'video': '3MSZA.mp4', 'caption': 'person flipped the light switch near the door.', 'timestamp': [24.296875, 30.40625]}
{'video': '3MSZA.mp4', 'caption': 'person turn the light switch on.', 'timestamp': [24.296875, 30.40625]}
```

我额外统计到：

- 当前 HF `test` 平均 query 词数约 `6.234`
- 和论文写的 `6.3` 很接近
- 当前 HF `test` 有 `1334` 个 unique videos
- 平均每个视频约 `2.789` 条 query

## 6. `lmms_eval/tasks/charades_sta` 在当前仓库里是怎样工作的

### 6.1 数据下载与视频解压

`charades.yaml` 里写的是：

```yaml
dataset_path: lmms-lab/charades_sta
dataset_kwargs:
  token: True
  cache_dir: charades_sta
  video: True
test_split: test
```

这意味着：

- `lmms_eval` 会从 HF 下载这个 dataset repo
- 因为 `video: True`，框架会额外把 repo 里的 zip / tar 媒体文件下载并解压到 `$HF_HOME/charades_sta`
- 对应逻辑在 `lmms_eval/api/task.py`

对于这个任务，repo 里的 `Charades_v1_480_part_*.zip` 会被解压出来，所以最终视频应当落在类似：

```text
$HF_HOME/charades_sta/Charades_v1_480/<video_name>.mp4
```

### 6.2 `doc_to_visual`

`utils.py` 中：

- 从样本读取 `doc["video"]`
- 拼成 `$HF_HOME/charades_sta/Charades_v1_480/<video>`
- 返回给模型作为输入视频

所以这个 benchmark 对模型的视觉输入是真实视频文件，而不是预抽特征。

### 6.3 `doc_to_text`

`doc_to_text` 会把 `caption` 包进一个固定 prompt，大意是：

- 在视频里找到这句事件描述对应的片段
- 输出格式要像 `The event happens in start - end`

因此当前 `lmms_eval` 不是像论文那样用 sliding windows + ranking + regression，而是：

- 把整段视频和一句 caption 一起喂给视频模型
- 让模型直接文本生成一对时间

### 6.4 `process_results`

预测结果会被保存成：

- key: `video>>>caption>>>timestamp`
- value: 模型原始输出字符串

这一步并不直接计算分数，而是先生成 submission 风格 JSON。

### 6.5 `eval_tvg.py`

真正的评测逻辑在 `eval_tvg.py`：

1. 从模型输出文本里抽时间
   - 支持 `24.3 - 30.4`
   - 支持分开说 `start ... end ...`
   - 支持 `HH:MM:SS`
   - 还有 fallback regex

2. 如果抽不出唯一时间段
   - 用一个明显错误的区间替代

3. 计算 IoU

4. 汇总：
   - `IoU@0.3`
   - `IoU@0.5`
   - `IoU@0.7`
   - `mIoU`

所以，当前 harness 的真实工作方式可以概括成：

`视频 + 句子 -> 生成字符串时间段 -> regex 抽取时间 -> IoU 评测`

## 7. 论文版 Charades-STA 与当前 `lmms_eval` 版 Charades-STA 的关键区别

这是最容易混淆、但最值得明确的一点。

### 论文在评什么

论文评的是一个专门为 TALL 设计的模型 family：

- 先滑窗采样 clip
- 再做跨模态对齐
- 再做时间边界回归

换句话说，论文的方法是“检索+回归式”的 temporal grounding。

### 当前 `lmms_eval` 在评什么

当前任务评的是通用视频大模型的“端到端生成式 grounding 能力”：

- 不显式构建滑窗
- 不显式输出 top-k proposals
- 不显式训练 CTRL 那种 alignment/regression 分支
- 而是直接让模型看视频后生成时间答案

所以：

- `benchmark` 还是 `Charades-STA`
- 但 `working mode` 已经从论文的 CTRL pipeline，变成了 `generative VLM evaluation`

这对理解结果非常重要，因为你不能把 `lmms_eval` 上一个视频模型的分数，简单理解成“复现了论文里的 CTRL”。

## 8. 我认为最值得记住的结论

1. `Charades-STA` 的本质是句子级视频时间定位，不是分类 benchmark。
2. 论文贡献分成两部分：新任务/新数据集 + CTRL 模型。
3. 论文里的关键技术点是“跨模态对齐 + 时间边界回归”，尤其强调 non-parameterized temporal regression 更有效。
4. 当前 `lmms_eval` 任务只借用了这个 benchmark 的数据形式，没有复现 CTRL 的训练范式。
5. 当前 HF 版本的数据规模与论文统计不完全一致：
   - 论文 test: `4233`
   - HF 当前 test: `3720`
6. 当前仓库中的 `charades_sta` 更像“submission-first”任务：
   - 主流程产出 JSON
   - 真正打分依赖 `eval_tvg.py`

## 9. 相关文件

- 论文源码：`~/.cache/nanochat/knowledge/1705.02101/egpaper_final.tex`
- task 配置：`lmms_eval/tasks/charades_sta/charades.yaml`
- task 逻辑：`lmms_eval/tasks/charades_sta/utils.py`
- 评测脚本：`lmms_eval/tasks/charades_sta/eval_tvg.py`

如果后面你愿意，我还可以继续做两件很直接的事：

1. 把 `charades_sta` 的当前评测流程画成一张简图  
2. 再进一步对比一下“论文原始 TALL/CTRL repo”和“本仓库 `lmms_eval` task”的差异点
