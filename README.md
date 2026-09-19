# MathGPT

一个面向算术推理的 GPT 风格语言模型。项目使用字符级 tokenizer 和 decoder-only Transformer，学习 `+`、`-`、`*`、`/` 四则运算，并通过 scratchpad（思维草稿）生成中间计算步骤后再给出结果。

## 特性

- 支持加法、减法、乘法和除法（含分数结果）。
- 支持多项式表达式、运算优先级和括号。
- **默认采用 1–6 年级课程训练**（算术 → 混合式 → 六年级混合式+方程；原一、二年级合并、原六、七年级方程并入六年级），由易到难逐级解锁，每级通过考试后才进入下一级。
- 使用逐位数字计算；推理格式上 prompt 与答案为正常十进制顺序，中间推理在 `<think>` 内完成（见 [`config.py`](config.py) 中 `reverse_in_think`）。
- 训练样本由程序动态生成；课程模式下每个年级、每个 epoch 都会重新采样，二元题 `(a, op, b)` 在单次训练过程中尽量不重复。
- 支持 CUDA、BF16/FP16 混合精度、梯度累积以及可选的 `torch.compile`。
- 默认 dropout 为 `0.0`，并使用 weight decay 进行正则化。

## 模型结构

模型是一个 GPT 风格的 decoder-only Transformer：

- `d_model=1024`
- `n_layer=24`
- `n_head=16`
- `d_ff=4096`
- 最大序列长度：`768`
- 词表 **120** token：数字、运算符、括号、等号、分号、`a–z`/`A–Z`、`<pad>`/`<bos>`/`<eos>`、思维链与 `<err>`，以及预留 **`<unuse1>`…`<unuse44>`**（见 `config.vocab_size`）
- token embedding 与输出层权重共享

默认配置约为 3 亿参数，适合在具有较大显存的 CUDA GPU 上训练。配置集中在 [`config.py`](config.py) 中。

## 环境要求

- Python 3.10+
- PyTorch 2.1+
- CUDA GPU（推荐训练时使用；CPU 可用于小规模 smoke test 或推理）

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 运行最小训练测试

`--smoke` 会使用较小的数据量和 batch size，适合先验证数据生成、模型前向、训练和评估流程：

```bash
python train.py --smoke --device cpu
```

如果使用 GPU：

```bash
python train.py --smoke --device cuda
```

### 正式训练（默认：课程 + 考试）

```bash
python train.py --device cuda
```

- 若不存在 `checkpoints/mathgpt.pt`：从**随机初始化**开始，从**一年级**练起。
- 若已有 checkpoint：自动**续训**（恢复权重、优化器/学习率调度器，以及课程进度）。
- 完全忽略旧 checkpoint、从头再来：

```bash
python train.py --fresh --device cuda
```

指定其它 checkpoint 路径：

```bash
python train.py --resume checkpoints/my_run.pt --device cuda
```

训练过程默认写入：

```text
checkpoints/mathgpt.pt
```

常用覆盖参数（课程各级题量在 [`curriculum.py`](curriculum.py)；`--smoke` 为快速试跑）：

```bash
python train.py --batch-size 16 --device cuda
```

## 训练策略

`train.py` 仅支持 [`curriculum.py`](curriculum.py) 定义的 **6 个年级**（1–5 算术与混合式，6 混合式+方程）。整体思路类似升年级：**只在当前年级的考试达到设定准确率后，才进入下一年级**；未通过则继续在本年级训练，直到通过或达到该年级 `max_epochs` 上限。

### 训练集与考试集的共通规则

- **题目形式**
  - **二元题**：`a op b=`（如 `37+8=`、`12*7=`），除法答案可为整数或既约分数 `分子/分母`。
  - **混合表达式**（四、五年级）：如 `(12+3)*4=`，含优先级与括号；训练目标在 `<think>` 内为逐步归约过程（见下文 scratchpad 节）。
  - **方程**（六年级，与表达式混合出题）：统一形如 **`a*x+k=c`**（`k=0` 时写作 `a*x=c`；`k<0` 时写作 `a*x-n`）。**输入不含末尾 `=`**（如 `3*x+5=14`），输出 **`x=3`**；若 **`a=0`**（如 `0*x+3=5`），输出 **`<err>`**。思维链为移项与除系数（见 scratchpad 节）。
- **数位采样**：先在 `[min_digits, max_digits]` 中均匀抽位数，再在该位数范围内均匀抽数值（上限不超过该级 `max_number`；乘法二元题操作数上限为 `mul_max_operand`）。
- **负数**：二元题中每个操作数独立以 `negative_fraction` 概率取负（表达式内的数由 `generate_expression_problems` 采样，不单独乘该系数）。
- **训练集**：每 epoch **独立随机**生成 `train_size` 道题（允许重复 `(a, op, b)` / 同一表达式或方程）。
- **考试集**：进入该年级时生成**一次**，整个年级内**同一批题**、多次 epoch 复考；**不**加入 `used_keys`，与训练集无 `(a, op, b)` 去重联动。
- **随机种子**（[`config.py`](config.py) / [`train.py`](train.py) / [`curriculum.py`](curriculum.py)）：
  - 训练：`seed + grade_id × 1_000_003 + grade_epoch × 97`
  - 考试：`curriculum_exam_seed_base + grade_id × 10_007`（默认 base = `900_001`）；表达式部分另用 `seed + 17`
- **通过标准**：考试集上**最终答案** exact match 准确率 ≥ 该级 `pass_accuracy`；至少训练 `min_epochs`（均为 1）后才开考。
- **监督格式**：默认 `reverse_in_think=True`，用户侧 prompt 为正常十进制；loss 仅对 `=` 之后（思维链 + 答案）计算。

---

### 一年级

| | 训练（每个 epoch） | 考试（固定题集，本年级全程复用） |
| --- | --- | --- |
| 题量 | 120 000 | 200 |
| 题型 | 100% 二元题 | 100% 二元题 |
| 运算 | `+`、`-`（均匀随机；含原一年级一位数加法） | 同左 |
| 数位 | 1–2 位，≤ 99 | 1–2 位，≤ 99 |
| 负数 | 约 5%（每个操作数独立） | 同左 |
| 混合表达式 | 无 | 无 |
| 通过线 / 最多 epoch | — | ≥ **98%**，最多 **60** epoch |

---

### 二年级

| | 训练（每个 epoch） | 考试（固定题集） |
| --- | --- | --- |
| 题量 | 150 000 | 200 |
| 题型 | 100% 二元题 | 100% 二元题 |
| 运算 | `+`、`-`、`*` | 同左 |
| 数位 | 1–2 位；乘法操作数 ≤ 99 | 同左 |
| 负数 | 约 10% | 同左 |
| 混合表达式 | 无 | 无 |
| 通过线 / 最多 epoch | — | ≥ **90%**，最多 **60** epoch |

---

### 三年级

| | 训练（每个 epoch） | 考试（固定题集） |
| --- | --- | --- |
| 题量 | 180 000 | 250 |
| 题型 | 100% 二元题 | 100% 二元题 |
| 运算 | `+`、`-`、`*`、`/` | 同左 |
| 数位 | 1–3 位，≤ 999 | 1–3 位，≤ 999 |
| 负数 | 约 15% | 同左 |
| 除法 | 非整除时为分数答案；`b=0` 为 `<err>`（生成时会跳过无效表达式，二元除法极少命中） | 同左 |
| 混合表达式 | 无 | 无 |
| 通过线 / 最多 epoch | — | ≥ **90%**，最多 **70** epoch |

---

### 四年级

| | 训练（每个 epoch） | 考试（固定题集） |
| --- | --- | --- |
| 题量 | 220 000 | 250 |
| 二元 / 表达式 | 约 **143 000** 二元 + **77 000** 表达式（`expression_fraction=0.35`） | 约 **162** 二元 + **88** 表达式 |
| 二元运算 | 四则均匀随机 | 同左 |
| 二元数位 | 2–4 位，≤ 9 999 | 考试：**2–4** 位（`exam_min/max_digits`） |
| 表达式 | 2–4 项；操作数 1–3 位；约 60% 带括号；内部运算符为 `+ - * /` 随机 | 项数/位数/括号比例与训练相同 |
| 负数（二元） | 约 20% | 同左 |
| 通过线 / 最多 epoch | — | ≥ **90%**，最多 **90** epoch |

---

### 五年级

| | 训练（每个 epoch） | 考试（固定题集） |
| --- | --- | --- |
| 题量 | 300 000 | 300 |
| 二元 / 表达式 | 约 **75 000** 二元 + **225 000** 表达式（`expression_fraction=0.75`） | 约 **75** 二元 + **225** 表达式 |
| 二元运算 | 四则 | 同左 |
| 二元数位 | 3–7 位，≤ 9 999 999 | 考试：**3–7** 位 |
| 表达式 | 4–10 项；操作数 3–7 位；约 75% 带括号 | 同训练参数 |
| 负数（二元） | 约 25% | 同左 |
| 通过线 / 最多 epoch | — | ≥ **90%**，最多 **120** epoch |

---

### 六年级

| | 训练（每个 epoch） | 考试（固定题集） |
| --- | --- | --- |
| 题量 | 300 000 | 300 |
| 表达式 / 方程 | 约 **225 000** 表达式 + **75 000** 方程（`expression_fraction=0.75`，`equation_fraction=0.25`） | 约 **225** 表达式 + **75** 方程 |
| 表达式 | **与五年级相同**：4–10 项；操作数 3–7 位；约 75% 带括号；四则 | 同训练参数 |
| 方程 | **`a*x+k=c`**；各半 **入门**（`\|k\|,|c|≤20`，`a∈[1,9]`，约 10% `a=0`）与 **提高**（`\|k\|,|c|≤99`，`a∈[1,12]`，约 15% `a=0`） | 同左 |
| 答案 | 表达式为整数/分数；方程为 `x=<整数>` 或 **`<err>`** | 考试 **exact match** |
| 通过线 / 最多 epoch | — | ≥ **90%**，最多 **90** epoch |

---

以上数值均来自 [`curriculum.py`](curriculum.py) 的 `default_curriculum()`。修改难度、题量、通过率或 epoch 上限请直接编辑该函数；`--smoke` 会按同结构缩小题量（每级 2 000 训练 / 40 考试）、降低通过线至 50%、每级最多 3 epoch。

### 单年级内的循环

1. **采样训练集**：按当前年级参数生成 **`train_size` 道题**（二元 / 混合式 / 方程）；种子随年级与 epoch 变化；每个 epoch 独立采样。
2. **训练一个 epoch**：与下文「优化与 batch」相同的前向、loss、反向与梯度累积。
3. **考试**（完成 `min_epochs` 后）：使用**固定种子**生成的 held-out 题集（题量见各级 `exam_size`），对**最终答案**做 exact match 准确率统计。
4. **晋级或重练**：准确率 ≥ 该级 `pass_accuracy` 则标记该年级通过并进入下一级；否则在本级继续下一个 epoch。
5. **保存 checkpoint**：每次考试后更新 `mathgpt.pt`；**通过该级考试**时还会写入 **`mathgpt_grade{N}.pt`**（保留「刚学完 N 年级」快照，默认开启，见 [`config.py`](config.py) 的 `save_grade_checkpoints`）。
6. **Ctrl+C**：训练或考试过程中按 Ctrl+C 会先写入 `checkpoints/mathgpt.pt`（含 `curriculum`、优化器状态与 `benchmark.interrupted`），再退出；直接再次运行 `python train.py` 从未完成的年级/epoch 接着练（epoch 内中断会回退 `grade_epoch`，重练该 epoch）。

`--smoke` 仍走同一套流程，但会使用缩小后的课程（更少样本、更低通过线、每级最多 3 个 epoch），用于快速验证管道。

### 优化与 batch（[`config.py`](config.py)）

- 优化器：AdamW，`lr=5e-5`，`weight_decay=0.05`，`betas=(0.9, 0.95)`。
- 微 batch `batch_size=16`，梯度累积 `grad_accum_steps=16`，**有效 batch ≈ 256**。
- **按年级分段学习率**：进入每个年级时重建 schedule——段长 = 该级 `max_epochs ×`（按该级 `train_size` 算的每 epoch 优化步数）；段内 **warmup + 余弦**，最低 `lr × lr_min_ratio`（默认 **0.3**）。**一年级且 global step 0** 用 `warmup_steps=1000`；**升年级 / 续训进入某级** 用 `grade_warmup_steps=400`。续训时根据 `current_grade` 与 `grade_epoch` 跳过段内已完成的步数；**不**恢复旧 checkpoint 里的 scheduler 状态。
- 梯度裁剪 `grad_clip=1.0`；默认 CUDA 上 **BF16** 混合精度（`--no-amp` 可关）。
- 默认只对 prompt 中等号 `=` **之后**的 token 计算 loss（`answer_only_loss=True`），包括思维链与最终答案。

### Checkpoint 里保存什么

路径默认为 `checkpoints/mathgpt.pt`。除模型权重与 `config` 外，课程训练还会保存：

| 字段 | 含义 |
| --- | --- |
| `step` | 全局训练步数 |
| `curriculum` | `passed_grades`、`current_grade`、`grade_epoch`、`exam_history`、`last_exam` |
| `benchmark` | 最近一次考试的准确率、是否通过、分桶统计等；若因 Ctrl+C 中断则含 `interrupted: true` |
| （额外文件） | 通过某级考试后另存 **`checkpoints/mathgpt_grade{N}.pt`**（与当时权重/优化器/课程进度一致，便于固定「学完 N 年级」的版本；主文件 `mathgpt.pt` 仍用于续训） |
| `optimizer` / `scheduler` / `scaler` | 可选（`save_optimizer_state=True` 时），用于无缝续训 |

再次运行 `python train.py` 会从未通过的最高年级接着练（checkpoint 须与当前 `config.chars` / 特殊 token 一致，否则请删除旧文件或使用 `--fresh`）。

全部 **6** 个年级通过后，再次运行会提示课程已完成；若要重新走课程请加 `--fresh`。旧版 checkpoint 的 `passed_grades` / `current_grade` 与当前年级编号可能不兼容，请用 **`--fresh`** 重开课程。

## 推理

对单个算式进行推理：

```bash
python infer.py --ckpt checkpoints/mathgpt.pt --prompt "123+45="
python infer.py --ckpt checkpoints/mathgpt.pt --prompt "12*7="
python infer.py --ckpt checkpoints/mathgpt.pt --prompt "3*x+5=14"
```

进入交互模式：

```bash
python infer.py --ckpt checkpoints/mathgpt.pt --interactive
```

也可以省略 `--interactive`，不提供 `--prompt` 时程序会自动进入 REPL。

## 评估

运行默认评估集：

```bash
python evaluate.py --ckpt checkpoints/mathgpt.pt
```

评估脚本会报告（均为最终答案 **exact match**）：

- 常规验证集（混合 `+` `-` `*`）
- 按数字位数统计的准确率（同上验证集）
- 较长数字 OOD：**`+` / `-` / `/`**
- 乘法（范围内与 OOD 长乘数）
- **除法**专项（`/`，答案可为整数或分数）
- **混合表达式**（项数/位数见 `config.expression_eval_*`）
- **一元一次方程** `a*x+k=c`（输出 `x=<整数>` 或 `a=0` 时 `<err>`）

可以通过参数控制各评估集大小：

```bash
python evaluate.py \
  --ckpt checkpoints/mathgpt.pt \
  --val-size 300 \
  --hard-size 200 \
  --mul-size 100 \
  --ood-mul-size 100 \
  --div-size 100 \
  --expr-size 100 \
  --eq-size 100
```

## 训练数据与 scratchpad

二元算式的输入形式类似：

```text
123+45=
12*7=
```

启用 scratchpad 时（默认 `reverse_in_think=True`），用户侧 prompt 为正常顺序，模型在思维标签内学习推导过程，例如：

```text
123+45=<think>123+45=168</think>168
```

混合表达式在思维草稿里保持**中缀形式**，按小学运算顺序（括号 → 同级 `*` `/` → 同级 `+` `-`）分阶段化简，步与步之间用 `=` 连接。同一阶段内会把当前层所有乘除（或所有加减）**一次性算完**，例如 `2*3+4*5` 一步到 `6+20`，而不是先 `6+4*5`。示例：

```text
2*3+4*5=<think>2*3+4*5=6+20=26</think>26
1+2*(3+4)=<think>1+2*(3+4)=1+2*7=1+14=15</think>15
(12+3)*4=<think>(12+3)*4=15*4=60</think>60
```

实现见 [`dataset.py`](dataset.py) 中的 `expression_infix_reduction_chain`。

**解方程**（六年级方程题）在思维链里用 **`;` 分步**（移项写成 `a*x=c-k`，再化简、除系数）；`a≠0` 时答案为 `x=…`，`a=0` 时为 `<err>`：

```text
3*x+5=14<think>3*x=14-5;3*x=9;x=9/3;x=3</think>x=3
0*x+3=5<think><err></think><err>
```

词表含 **`a–z` / `A–Z`** 与 **`<err>`**（[`config.py`](config.py)）；未知数为小写 **`x`**。考试与推理要求最终输出与标答 **`x=<整数>`** 或 **`<err>`** 完全一致。

训练时 loss 从 **`<think>` 起**（用户侧 prompt 为 `3*x+5=14`，不含末尾 `=`；式中 `=` 仅作等号，不参与算术题那种「prompt 尾 `=`」切分）；监督思维链与最终答案。

## 目录说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 模型、数据和训练超参数 |
| `curriculum.py` | 1–6 年级课程（含混合式与方程）、考试题生成与进度结构 |
| `model.py` | GPT 风格 Transformer 模型 |
| `dataset.py` | 算式生成、scratchpad 构造和数据集 |
| `tokenizer.py` | 字符级 tokenizer |
| `train.py` | 课程训练、年级考试与 checkpoint 保存 |
| `evaluate.py` | 基准评估 |
| `infer.py` | 单次推理和交互式推理 |
| `requirements.txt` | Python 依赖 |

## 注意事项

1. 单字符词表在 `config.chars`；多字符 token 为 `active_specials`（思维链、`<err>`）加预留 `reserved_specials`（`<unuseN>` 填至 `vocab_size=120`）。修改 `chars` / `vocab_size` 后须重新训练，旧 checkpoint 不会自动迁移。
2. 修改模型宽度、层数、头数或 FFN 大小后，也不能加载结构不匹配的 checkpoint。
3. 请以**各级考试准确率**和 checkpoint 中的 `benchmark` 为准，不要只看训练 loss。
4. 默认最大序列长度为 `768`；较长表达式可能被截断或超出模型上下文限制。
5. `evaluate.py` 和 `infer.py` 使用 `dropout=0.0` 进行确定性推理。

## 许可证

当前仓库未声明开源许可证。如需公开发布，建议根据你的使用场景补充 LICENSE 文件。
