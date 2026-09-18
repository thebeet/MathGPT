# MathGPT

一个面向算术推理的轻量级 GPT 风格语言模型。项目使用字符级 tokenizer 和 decoder-only Transformer，学习 `+`、`-`、`*` 运算，并通过 scratchpad（思维草稿）生成中间计算步骤后再给出结果。

## 特性

- 支持加法、减法和乘法。
- 支持多项式表达式、运算优先级和括号。
- 使用逐位数字计算，默认采用低位在前（LSD-first）的数字表示，帮助模型学习进位、借位和乘法过程。
- 训练样本由程序动态生成，每个 epoch 生成新的样本，并将训练集与验证集分离。
- 支持普通验证、较长数字 OOD（分布外）验证和乘法专项验证。
- 支持 CUDA、BF16/FP16 混合精度、梯度累积以及可选的 `torch.compile`。
- 默认 dropout 为 `0.0`，并使用 weight decay 进行正则化。

## 模型结构

模型是一个 GPT 风格的 decoder-only Transformer：

- `d_model=1024`
- `n_layer=24`
- `n_head=16`
- `d_ff=4096`
- 最大序列长度：`768`
- 词表：数字、运算符、括号、等号、分号和特殊 token
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

### 正式训练

```bash
python train.py --device cuda
```

常用覆盖参数：

```bash
python train.py \
  --epochs 10 \
  --batch-size 16 \
  --train-size 1000000 \
  --device cuda
```

训练完成后，最佳 checkpoint 默认保存为：

```text
checkpoints/mathgpt.pt
```

训练期间会根据验证集准确率和 loss 保存最佳模型。若需要调整学习率、模型规模、数据范围、dropout 或 OOD 设置，请直接修改 [`config.py`](config.py)。

## 推理

对单个算式进行推理：

```bash
python infer.py --ckpt checkpoints/mathgpt.pt --prompt "123+45="
python infer.py --ckpt checkpoints/mathgpt.pt --prompt "12*7="
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

评估脚本会报告：

- 常规验证集准确率
- 按数字位数统计的准确率
- 较长数字的加减法 OOD 准确率
- 乘法准确率
- 较长乘数的 OOD 准确率

可以通过参数控制各评估集大小：

```bash
python evaluate.py \
  --ckpt checkpoints/mathgpt.pt \
  --val-size 300 \
  --hard-size 200 \
  --mul-size 100 \
  --ood-mul-size 100
```

## 训练数据与 scratchpad

二元算式的输入形式类似：

```text
123+45=
12*7=
```

启用 scratchpad 时，模型学习的目标大致为：

```text
123+45=<think>123+45=168</think>168
```

多项式表达式会先按照运算优先级转换为 postfix，并生成逐步归约过程。例如：

```text
(12+3)*4=<think>12+3=15;15*4=60</think>60
```

训练时默认只对等号之后的答案和 scratchpad 部分计算 loss（`answer_only_loss=True`）。

## 目录说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 模型、数据和训练超参数 |
| `model.py` | GPT 风格 Transformer 模型 |
| `dataset.py` | 算式生成、scratchpad 构造和数据集 |
| `tokenizer.py` | 字符级 tokenizer |
| `train.py` | 训练、验证和 checkpoint 保存 |
| `evaluate.py` | 基准评估 |
| `infer.py` | 单次推理和交互式推理 |
| `requirements.txt` | Python 依赖 |

## 注意事项

1. 修改字符集或特殊 token 会改变词表大小，旧 checkpoint 可能无法直接加载。
2. 修改模型宽度、层数、头数或 FFN 大小后，也不能直接加载结构不匹配的 checkpoint。
3. 训练使用动态生成数据，建议同时关注验证集准确率和 OOD 准确率，不要只看训练 loss。
4. 默认最大序列长度为 `768`；较长表达式可能被截断或超出模型上下文限制。
5. `evaluate.py` 和 `infer.py` 使用 `dropout=0.0` 进行确定性推理。

## 许可证

当前仓库未声明开源许可证。如需公开发布，建议根据你的使用场景补充 LICENSE 文件。
