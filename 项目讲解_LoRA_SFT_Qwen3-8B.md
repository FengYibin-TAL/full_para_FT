# 基于 Qwen3-8B 的关键词抽取 LoRA 监督微调项目讲解

本文档面向课程报告、项目说明或实验总结场景，围绕 `full_para_FT` 目录中的 **LoRA 训练与推理 Notebook** 展开。文档从项目介绍开始，依次说明需求、技术方案、实现过程、测试方法与常见问题，而不是直接从某个中间实现章节切入。

项目对应文件主要为：

- `sft-LoRA-train.ipynb`
- `sft-LoRA-inference.ipynb`
- `data/keywords_data_train.jsonl`
- `data/keywords_data_test.jsonl`

---

## 1 项目介绍

### 1.1 项目背景

在自然语言处理任务中，**关键词抽取**是文本理解、信息检索、知识图谱构建、摘要生成等任务的重要基础步骤。传统关键词抽取方法通常依赖规则、统计特征或独立的分类模型，而随着大语言模型的发展，越来越多的场景开始采用**指令微调**方式，让模型直接根据提示生成结构化结果。

本项目选择 **Qwen/Qwen3-8B** 作为基座模型，通过 **LoRA（Low-Rank Adaptation）** 方法进行监督微调，使模型能够针对“给定标题和正文，输出关键词”的任务进行更稳定、更符合标注风格的生成。

### 1.2 项目目标

本项目的目标包括：

1. 使用对话式数据对 `Qwen/Qwen3-8B` 进行关键词抽取任务的监督微调。
2. 采用 **LoRA** 而非全参微调，降低训练参数量与训练门槛。
3. 给出从环境准备、数据处理、模型训练到推理验证的完整流程。
4. 将 LoRA 权重与推理流程整理为可复现的 Notebook 实现。

### 1.3 项目特点

与同目录中的 `sft-helloworld-train.ipynb` 不同，本项目：

- 使用的是 **8B 规模基座模型**，能力更强，但显存压力也更大。
- 使用 **PEFT + LoRA**，训练的并不是全部模型参数，而是注入到指定线性层中的低秩适配器参数。
- 在推理时既可以保留“基座 + LoRA adapter”的形式，也可以通过 `merge_and_unload()` 将 LoRA 权重合并回基座，得到一个单独可部署的模型目录。

### 1.4 项目目录结构

`full_para_FT` 目录中与 LoRA 相关的主要内容如下：

| 路径 | 说明 |
|------|------|
| `data/keywords_data_train.jsonl` | 训练集，JSONL 格式 |
| `data/keywords_data_test.jsonl` | 验证/测试集，JSONL 格式 |
| `sft-LoRA-train.ipynb` | LoRA 训练 Notebook |
| `sft-LoRA-inference.ipynb` | LoRA 推理与合并模型 Notebook |
| `README.md` | 项目总说明 |

---

## 2 需求分析

### 2.1 业务需求

本项目要解决的问题是：给定一段中文学术或技术文本，让模型输出该文本的关键词。

输入通常包含：

- 指令，例如“抽取出文本中的关键词”
- 标题
- 正文

输出通常为：

- 一串关键词文本
- 风格与训练标签保持一致
- 不额外输出解释、分析过程或无关内容

### 2.2 数据需求

训练数据采用 **对话式 JSONL** 格式，每一行是一条样本，核心字段是 `conversation`。每条 `conversation` 中包含多轮 `human` 与 `assistant` 内容。

这种设计的优点是：

1. 与聊天模型的训练格式天然兼容。
2. 可直接映射为 `messages=[{"role": ..., "content": ...}]` 的结构。
3. 便于后续使用模型原生 chat template 组织输入。

### 2.3 技术需求

为了完成本项目，需要满足以下技术条件：

| 项目 | 要求 |
|------|------|
| 模型 | `Qwen/Qwen3-8B` |
| 微调方式 | LoRA |
| 训练框架 | `transformers` + `trl` + `peft` |
| 数据处理 | `datasets` |
| 设备 | 推荐使用较大显存 GPU |
| 日志可视化 | `tensorboard`（可选） |

### 2.4 约束与难点

LoRA 虽然降低了可训练参数量，但并不等于显存消耗会大幅下降。对于 8B 级模型，训练时依然要关注：

- `per_device_train_batch_size`
- 序列长度
- 混合精度
- 梯度累积
- 是否启用梯度检查点

另外，`peft` 的安装顺序与 Notebook 内核状态也会影响训练流程的稳定性。如果先 import 了 `trl` 再安装 `peft`，可能会出现各种导入与保存错误，因此本项目特别强调：**安装依赖后重启内核，再顺序执行 Notebook。**

---

## 3 技术方案

### 3.1 为什么选择 Qwen3-8B

`Qwen3-8B` 是中大规模开源中文/多语言大模型，具有较好的中文理解与生成能力。对于关键词抽取这类需要一定语义概括能力的任务，8B 模型相比更小模型通常具有更高的上限。

### 3.2 为什么选择 LoRA

如果直接对 8B 模型做全参微调，训练成本与显存成本都较高，因此本项目采用 LoRA：

1. 通过低秩矩阵近似的方式，只训练少量新增参数。
2. 基座模型主体权重可以保持不变。
3. 更适合单卡实验环境。
4. 训练完成后既可以独立保存 adapter，也可以合并导出。

### 3.3 本项目整体流程

本项目整体流程如下：

1. 设置 Hugging Face 下载环境与缓存目录。
2. 安装 `peft`、`trl`、`datasets` 等依赖。
3. 重启 Notebook 内核，确保后续模块导入干净。
4. 加载 `Qwen/Qwen3-8B` 和分词器。
5. 加载 JSONL 数据集并转换为 `messages` 格式。
6. 定义 `LoraConfig` 与 `SFTConfig`。
7. 使用 `SFTTrainer(..., peft_config=...)` 开始训练。
8. 保存 LoRA adapter 到固定目录。
9. 推理阶段加载基座模型与 adapter。
10. 通过 `merge_and_unload()` 合并模型，完成生成与导出。

---

## 4 项目实现

本章对应 `sft-LoRA-train.ipynb` 与 `sft-LoRA-inference.ipynb` 的核心实现。

### 4.1 环境准备

训练 Notebook 首先设置 Hugging Face 镜像与缓存目录，并安装依赖。

```python
%env HF_ENDPOINT=https://hf-mirror.com
# %env HF_ENDPOINT=https://huggingface.co
%env HF_HOME=/root/autodl-tmp/hf

# LoRA 依赖 PEFT。安装完成后请 Kernel → Restart，再运行后续单元。
!pip install peft trl datasets
```

这段代码的作用如下：

- `HF_ENDPOINT`：指定 Hugging Face 镜像地址，便于在国内环境中加速下载模型。
- `HF_HOME`：指定模型缓存目录，避免默认缓存写入系统盘。
- `!pip install peft trl datasets`：安装 LoRA 微调所需的核心依赖。

这里最关键的一点是：**安装完成后必须重启内核**。原因在于，如果在未安装 `peft` 时已经导入过 `trl` 或 `transformers`，后续即使安装成功，也可能因为模块缓存导致：

- `NameError: get_peft_model`
- `NameError: PeftModel`
- `PicklingError: SchedulerType`

因此，正确流程是：

1. 运行安装单元
2. 重启内核
3. 从头重新执行后续单元

### 4.2 加载基座模型与分词器

训练所使用的基座模型为 `Qwen/Qwen3-8B`，代码如下：

```python
# 加载model和Tokenizer
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

model_name = 'Qwen/Qwen3-8B'
model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

该部分完成的工作包括：

- 从 Hugging Face Hub 下载或读取缓存中的 8B 模型权重。
- 使用 `AutoTokenizer` 加载与模型匹配的分词器。
- 将模型加载为 `bfloat16`，减少显存占用。

这里的 `dtype=torch.bfloat16` 很重要，它说明训练阶段使用的是 **bf16 混合精度**。如果硬件不支持 `bf16`，则需要根据环境调整为其他精度策略。

### 4.3 数据集加载与格式转换

本项目训练数据位于：

- `data/keywords_data_train.jsonl`
- `data/keywords_data_test.jsonl`

对应代码如下：

```python
# 数据集
from datasets import load_dataset
dataset_dict = load_dataset('json',data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})

def map_func(example):
    conversation = example['conversation']
    messages = []
    for item in conversation:
        messages.append({'role':'user','content':item['human']})
        messages.append({'role':'assistant','content':item['assistant']})
    return {'messages':messages}

dataset_dict = dataset_dict.map(map_func,batched=False,remove_columns=['dataset','conversation','category','conversation_id'])
```

这部分实现可以分为两步理解。

第一步，使用 `load_dataset('json', ...)` 将本地 JSONL 文件读入为 Hugging Face Dataset：

- `train` 对应训练集
- `test` 对应验证/测试集

第二步，定义 `map_func(example)`，将原始的 `conversation` 字段转为 `messages`：

- `human` → `{"role": "user", "content": ...}`
- `assistant` → `{"role": "assistant", "content": ...}`

转换后，样本格式就适配了聊天模型微调流程。

`remove_columns` 的作用是去掉后续训练不再需要的字段：

- `dataset`
- `conversation`
- `category`
- `conversation_id`

这样能减少数据冗余，也让 `SFTTrainer` 的输入结构更清晰。

### 4.4 导入 SFTTrainer

在确认 `peft` 已经安装并且内核已重启之后，才能安全导入：

```python
# 流程：先运行上一格的 pip install，然后务必「重启内核」，再从上到下执行本 Notebook。
# 原因：若曾在未安装 peft 时 import 过 trl，用 importlib.reload(transformers...) 虽能临时修
# NameError，但会在保存 checkpoint 时触发 PicklingError（SchedulerType 等枚举不是同一对象）。
from trl import SFTConfig, SFTTrainer
```

这里的注释并不是多余说明，而是 LoRA 训练能否正常进行的关键前提。它明确了：

- 不要用 `reload(transformers)` 这类热修方式。
- 安装依赖后一定要让内核进入一个“干净状态”。

### 4.5 LoRA 配置与训练参数配置

这是 LoRA 训练的核心部分：

```python
from peft import LoraConfig

import os

# TensorBoard：Transformers 5.x 用 TENSORBOARD_LOGGING_DIR；logging_dir 参数已废弃且不会写入该路径
# 必须在实例化 SFTTrainer 之前设置，以便 TensorBoardCallback 读到
os.environ["TENSORBOARD_LOGGING_DIR"] = "/root/tf-logs"

# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 4
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 8
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)


# Configure trainer
training_args = SFTConfig(
    output_dir="/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    report_to="tensorboard",
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    bf16=True,
    warmup_steps=50
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
    peft_config=peft_config
)
```

这一部分可以拆成三层理解。

#### 4.5.1 TensorBoard 日志目录

```python
os.environ["TENSORBOARD_LOGGING_DIR"] = "/root/tf-logs"
```

在当前 Transformers 版本中，TensorBoard 的日志目录不建议再只依赖老的 `logging_dir` 参数，而是直接通过环境变量指定。这样在训练过程中，`report_to="tensorboard"` 才能把事件日志真正写入 `/root/tf-logs`。

#### 4.5.2 LoRA 参数

```python
rank_dimension = 4
lora_alpha = 8
lora_dropout = 0.05
```

它们的含义分别是：

- `r`：LoRA 的低秩维度，越小参数量越少。
- `lora_alpha`：LoRA 分支的缩放系数。
- `lora_dropout`：LoRA 路径上的 dropout。

在 `LoraConfig` 中：

- `bias="none"` 表示不训练 bias。
- `target_modules="all-linear"` 表示对模型中的线性层统一注入 LoRA。
- `task_type="CAUSAL_LM"` 说明当前任务是因果语言建模。

#### 4.5.3 训练参数

`SFTConfig` 中最重要的参数包括：

- `output_dir`：训练输出目录
- `max_steps=1000`：最大训练步数
- `per_device_train_batch_size=4`：单卡 batch size
- `learning_rate=5e-5`
- `logging_steps=10`
- `save_steps=100`
- `eval_steps=100`
- `load_best_model_at_end=True`
- `bf16=True`
- `warmup_steps=50`

训练器初始化中传入：

- `model`
- `args`
- `train_dataset`
- `eval_dataset`
- `processing_class=tokenizer`
- `peft_config=peft_config`

其中 `peft_config` 是最关键的一项，它告诉 `SFTTrainer`：这不是普通全参训练，而是要对模型做 LoRA 包装。

### 4.6 训练前样本检查

在正式训练前，Notebook 中还加了一段自检代码：

```python
# 察看数据集处理结果
dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))
print(tokenizer.decode(batch['input_ids'][0]))
```

这段代码的意义在于：

1. 确认 `messages` 已经被正确编码。
2. 确认 `SFTTrainer` 拼接后的输入文本格式符合预期。
3. 提前发现模板异常、标签错位、分词异常等问题。

### 4.7 模型训练与保存

训练与保存代码非常简洁：

```python
trainer.train()
```

```python
trainer.save_model('/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/best')
```

`trainer.train()` 会根据前面定义的 `SFTConfig`：

- 执行训练循环
- 定期 logging
- 定期 evaluate
- 定期保存 checkpoint

而 `trainer.save_model(...)` 会将最终可用于推理的 LoRA 结果另存到固定目录：

- `/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/best`

这个目录在后面的推理 Notebook 中会直接用到。

---

## 5 推理流程实现

LoRA 推理代码位于 `sft-LoRA-inference.ipynb`。

### 5.1 推理环境准备

```python
%env HF_ENDPOINT=https://hf-mirror.com
%env HF_HOME=/root/autodl-tmp/hf

!pip install peft
```

这里与训练阶段类似，主要是确保推理环境具备 `peft`，以及基座模型仍能通过镜像正常下载。

### 5.2 加载基座模型与 LoRA 适配器

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16
)

peft_model = PeftModel.from_pretrained(
    base_model, "/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/best", dtype=torch.float16
)

merged_model = peft_model.merge_and_unload()
```

这段代码完成了三件事：

1. 加载原始 `Qwen/Qwen3-8B` 基座。
2. 从 `best` 目录中加载 LoRA adapter。
3. 通过 `merge_and_unload()` 合并回基座，得到一个独立可推理的模型对象。

这里使用 `dtype=torch.float16`，是推理侧更常见的选择。

### 5.3 构造输入并生成结果

```python
# prepare the model input
prompt = "抽取出文本中的关键词：\n标题：人工神经网络在猕猴桃种类识别上的应用\n文本：在猕猴桃介电特性研究的基础上,将人工神经网络技术应用于猕猴桃的种类识别.该种类识别属于模式识别,其关键在于提取样品的特征参数,在获得特征参数的基础上,选取合适的网络通过训练来进行识别.猕猴桃种类识别的研究为自动化识别果品的种类、品种和新鲜等级等提供了一种新方法,为进一步研究果品介电特性与其内在品质的关系提供了一定的理论与实践基础."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
print(text)
```

这一步利用 tokenizer 的 chat template，把普通 prompt 转换成模型期望的对话格式。

之后执行生成：

```python
model_inputs = tokenizer([text], return_tensors="pt").to(merged_model.device)

# conduct text completion
generated_ids = merged_model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```

这里的重点是：

- `generate(...)` 得到完整输出 token。
- 通过切片去掉输入部分。
- 再尝试根据特殊 token `151668`（对应 `</think>`）把“思考内容”和“最终内容”分开。

最终真正要关注的通常是：

- `content`

它才是关键词抽取结果。

### 5.4 导出合并后的模型

如果需要把 LoRA 合并后的模型作为一个独立模型目录保存下来，可以执行：

```python
merged_model.save_pretrained("/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/merged")
tokenizer.save_pretrained("/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/merged")
```

执行后会得到：

- `/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/merged`

后续可以直接从这个目录加载模型和 tokenizer，而无需再次挂载 LoRA adapter。

---

## 6 测试与运行说明

### 6.1 训练测试

运行 `sft-LoRA-train.ipynb` 时，应按如下顺序：

1. 运行环境与安装单元
2. **重启内核**
3. 从头顺序运行 Notebook
4. 观察训练日志、评估日志与保存目录

训练完成后，重点检查：

- `output_dir` 是否生成 checkpoint
- `best` 目录是否存在
- `/root/tf-logs` 下是否有 `tfevents`

### 6.2 推理测试

运行 `sft-LoRA-inference.ipynb` 后，应验证：

1. 模型能正常加载基座与 adapter
2. `merge_and_unload()` 不报错
3. `generate()` 可以返回关键词结果
4. `merged` 目录可正常保存

### 6.3 训练产物

本项目训练后常见产物包括：

| 路径 | 作用 |
|------|------|
| `/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA` | 训练 checkpoint 根目录 |
| `/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/best` | 最终 adapter 目录 |
| `/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/merged` | 合并后的完整模型目录 |
| `/root/tf-logs` | TensorBoard 日志目录 |

---

## 7 常见问题分析

### 7.1 `ModuleNotFoundError: No module named 'peft'`

原因：未安装 `peft`。  
处理：运行 `!pip install peft`，然后**重启内核**。

### 7.2 `NameError: get_peft_model` 或 `PeftModel`

原因：在安装 `peft` 之前已经导入了 `trl` / `transformers`，导致内部模块状态不一致。  
处理：**不要 reload，直接重启内核，从头执行。**

### 7.3 `PicklingError: SchedulerType`

原因：使用 `importlib.reload(transformers...)` 等方式热重载模块，导致训练参数中的枚举对象与当前模块定义不一致。  
处理：重启内核，不再使用 reload 补丁。

### 7.4 `CUDA out of memory`

原因：8B 模型即使使用 LoRA，训练前向仍然很占显存。  
处理方向：

- 减小 `per_device_train_batch_size`
- 设置 `max_length`
- 开启 `gradient_checkpointing`
- 使用 `gradient_accumulation_steps`

### 7.5 TensorBoard 没有曲线

原因通常包括：

- 未设置 `report_to="tensorboard"`
- 未设置 `TENSORBOARD_LOGGING_DIR`
- 没有生成 `tfevents`

检查方式：

- 看 `/root/tf-logs` 是否存在事件文件
- 启动 `tensorboard --logdir /root/tf-logs`

---

## 8 总结

本项目基于 **Qwen3-8B + PEFT LoRA + TRL SFTTrainer**，完成了一个面向关键词抽取任务的监督微调流程。整个项目从工程角度上可以概括为：

1. 用 `datasets` 加载对话式 JSONL 数据；
2. 将数据映射为 `messages`；
3. 用 `LoraConfig` 指定低秩适配器训练策略；
4. 用 `SFTConfig` 指定训练、评估、保存和日志参数；
5. 用 `SFTTrainer` 完成 LoRA 微调；
6. 在推理阶段通过 `PeftModel.from_pretrained` 加载 adapter，并通过 `merge_and_unload()` 生成可独立部署的合并模型。

相较于全参微调，LoRA 更适合大模型场景下的实验与快速验证；但需要注意的是，LoRA 主要减少的是**可训练参数量**，并不能完全解决 8B 模型前向阶段的显存压力。因此，在实际训练中仍需要结合 batch size、序列长度和混合精度策略进行综合调优。
# 基于 Qwen3-8B 的关键词抽取 LoRA 监督微调实现

任务：在对话式 JSONL 数据上，对通义千问 **Qwen/Qwen3-8B** 做 **LoRA（PEFT）+ TRL SFT**。实现载体为 **`full_para_FT` 目录**下的 Jupyter Notebook（非多 `.py` 分包），下文按**逻辑模块**对应到各代码单元，颗粒度与「配置块 + 完整代码清单」类文档一致。

---

## 3.4.1 环境准备

在 **`full_para_FT`** 目录下，LoRA 相关代码按下列**逻辑模块**组织（与 `sft-LoRA-train.ipynb`、`sft-LoRA-inference.ipynb` 中的执行顺序一致）：

 **环境与依赖单元**：设置 Hugging Face 镜像与缓存目录；安装 `peft`、`trl`、`datasets`；**安装完成后须重启内核**再继续（避免未装 `peft` 时已 `import trl` 导致后续异常；勿用 `importlib.reload(transformers...)` 硬修，否则保存 checkpoint 时可能出现 `PicklingError`）。

 **基座模型与分词器**：加载 `Qwen/Qwen3-8B` 与 `AutoTokenizer`，dtype 使用 `bfloat16`（需 GPU 支持；与推理侧 dtype 策略宜一致）。

 **数据处理**：`load_dataset('json', ...)` 读取 `data/keywords_data_*.jsonl`，`map` 为 TRL 所需的 `messages` 格式。

 **TRL 导入**：`from trl import SFTConfig, SFTTrainer`（在 pip 安装并重启内核之后执行）。

 **LoRA 与训练配置**：`LoraConfig`（秩、alpha、目标模块等）+ `SFTConfig`（输出目录、步数、batch、TensorBoard 等）+ `SFTTrainer(..., peft_config=...)`。

 **可选自检**：`get_train_dataloader()` 取 batch，`decode` 查看拼接样本。

 **训练与保存**：`trainer.train()`；`trainer.save_model(.../best)` 写出适配器目录。

 **推理 Notebook**：基座 + `PeftModel.from_pretrained` 加载 `best`；`merge_and_unload()`；`apply_chat_template` + `generate`；可选 `save_pretrained` 导出合并模型到 `merged`。

**依赖概要**：Python 3.10+、`torch`（CUDA）、`transformers`、`trl`、`datasets`、`accelerate`、**`peft`**；可选 `tensorboard`（与 `report_to="tensorboard"` 配合）。

---

## 3.4.2 代码清单

以下代码均来自仓库内 **`sft-LoRA-train.ipynb`**、**`sft-LoRA-inference.ipynb`**（路径以示例机器为准，换环境请整体替换根目录）。注释为讲解补充，便于对照「每一行在做什么」。

### 1）环境与依赖安装（`sft-LoRA-train.ipynb` — 首个代码单元）

```python
# ------------ Hugging Face 镜像与缓存 ------------
# HF_ENDPOINT：国内常用镜像端点，无法直连 huggingface.co 时使用
%env HF_ENDPOINT=https://hf-mirror.com
# HF_HOME：模型与数据集缓存根目录，建议指向数据盘（需写权限）
%env HF_HOME=/root/autodl-tmp/hf

# ------------ Python 依赖 ------------
# peft：LoRA 等参数高效微调
# trl：SFTTrainer / SFTConfig
# datasets：load_dataset 读取 jsonl
# 安装完成后务必执行：Kernel -> Restart，再运行后续单元
!pip install peft trl datasets
```

---

### 2）加载基座模型与分词器（`sft-LoRA-train.ipynb`）

```python
# 从 Transformers 加载因果语言模型与分词器
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 基座模型 ID（Hub）；8B 体积大，首次会下载到 HF_HOME 缓存
model_name = "Qwen/Qwen3-8B"
# dtype=bfloat16：降低权重占用；需 GPU 支持 bf16
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

### 3）数据加载与 `messages` 映射（`sft-LoRA-train.ipynb`）

```python
# datasets：按文件路径加载 JSONL，注册 train / test 两个 split
from datasets import load_dataset

dataset_dict = load_dataset(
    "json",
    data_files={
        "train": "data/keywords_data_train.jsonl",
        "test": "data/keywords_data_test.jsonl",
    },
)


def map_func(example):
    """将样本中的多轮 human/assistant 转为 OpenAI 风格 messages。"""
    conversation = example["conversation"]
    messages = []
    for item in conversation:
        messages.append({"role": "user", "content": item["human"]})
        messages.append({"role": "assistant", "content": item["assistant"]})
    return {"messages": messages}


# batched=False：逐条映射；去掉与训练无关的列，减小内存
dataset_dict = dataset_dict.map(
    map_func,
    batched=False,
    remove_columns=["dataset", "conversation", "category", "conversation_id"],
)
```

---

### 4）导入 TRL（`sft-LoRA-train.ipynb`；须在 pip 安装 peft 并重启内核后执行）

```python
# 若曾在未安装 peft 时 import 过 trl，不要用 importlib.reload(transformers...) 修补——
# 可能在保存 checkpoint 时触发 PicklingError（如 SchedulerType 枚举不一致）
from trl import SFTConfig, SFTTrainer
```

---

### 5）`LoraConfig`、`SFTConfig` 与 `SFTTrainer`（`sft-LoRA-train.ipynb`）

```python
from peft import LoraConfig
import os

# ------------ TensorBoard 日志目录（Transformers 5.x）------------
# 通过环境变量指定；勿仅依赖已废弃的 TrainingArguments.logging_dir
os.environ["TENSORBOARD_LOGGING_DIR"] = "/root/tf-logs"

# ------------ LoRA 超参数 ------------
# r：低秩矩阵的秩，越小参数量越少，通常 4~64 之间尝试
rank_dimension = 4
# lora_alpha：对 LoRA 输出的缩放，常与 r 成比例（如 2*r）
lora_alpha = 8
# lora_dropout：LoRA 路径上的 dropout，抑制过拟合
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",  # 是否训练 bias；none 表示不训练
    target_modules="all-linear",  # 在哪些线性层上注入 LoRA；可按模型结构调整
    task_type="CAUSAL_LM",  # 因果语言模型任务类型（PEFT 用）
)

# ------------ SFT 训练参数 ------------
training_args = SFTConfig(
    # output_dir：checkpoint、trainer_state 等输出根目录
    output_dir="/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA",
    max_steps=1000,  # 优化步数上限
    per_device_train_batch_size=4,  # 单卡 batch；显存不足可减小并配合 gradient_accumulation_steps
    learning_rate=5e-5,
    logging_steps=10,
    report_to="tensorboard",  # 写入 TensorBoard；需已安装 tensorboard 且配置 TENSORBOARD_LOGGING_DIR
    save_steps=100,
    save_total_limit=2,  # 最多保留 checkpoint 份数
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    bf16=True,  # 混合精度；与模型加载侧 bf16 一致
    warmup_steps=50,
    # 显存仍紧张时可增加：max_length=512, gradient_checkpointing=True,
    # gradient_accumulation_steps=2 等（按 TRL 版本字段名为准）
)

# SFTTrainer 在 peft_config 非空时会内部调用 get_peft_model 包装基座
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)
```

---

### 6）训练前自检（可选，`sft-LoRA-train.ipynb`）

```python
# 取一个 batch，解码第一条样本，确认 chat 模板与数据是否符合预期
dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))
print(tokenizer.decode(batch["input_ids"][0]))
```

---

### 7）训练与保存适配器（`sft-LoRA-train.ipynb`）

```python
# 进入训练循环：按 SFTConfig 打日志、评估、存 checkpoint
trainer.train()

# 将当前模型（LoRA 包装后）另存到固定目录，供推理时 PeftModel.from_pretrained 加载
trainer.save_model("/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/best")
```

---

### 8）推理：环境、基座、挂载 LoRA、合并（`sft-LoRA-inference.ipynb`）

```python
# ------------ 环境与 peft（推理环境若未装 peft 需先执行）------------
%env HF_ENDPOINT=https://hf-mirror.com
%env HF_HOME=/root/autodl-tmp/hf
!pip install peft

# ------------ 加载基座与分词器 ------------
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 推理侧使用 float16；需与显卡兼容
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
)

# 在基座上加载训练保存的 LoRA 权重目录（adapter）
peft_model = PeftModel.from_pretrained(
    base_model,
    "/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/best",
    dtype=torch.float16,
)

# 将 LoRA 权重合并进基座，得到单模型，便于直接 generate 或再 save_pretrained
merged_model = peft_model.merge_and_unload()
```

---

### 9）构造提示词与生成（`sft-LoRA-inference.ipynb`）

```python
# 业务 prompt：与训练分布越接近越好
prompt = (
    "抽取出文本中的关键词：\n标题：人工神经网络在猕猴桃种类识别上的应用\n"
    "文本：在猕猴桃介电特性研究的基础上,将人工神经网络技术应用于猕猴桃的种类识别..."
)
messages = [{"role": "user", "content": prompt}]
# Qwen3 聊天模板；enable_thinking=False 关闭思考链模式（按模型支持情况）
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
print(text)

model_inputs = tokenizer([text], return_tensors="pt").to(merged_model.device)

# max_new_tokens 过大浪费显存与时间；关键词任务可改为数百以内
generated_ids = merged_model.generate(
    **model_inputs,
    max_new_tokens=32768,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

# 按 Qwen 特殊 token 切分 thinking / 正文（token id 以当前 tokenizer 为准）
try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
print("thinking content:", thinking_content)
print("content:", content)
```

---

### 10）可选：导出合并后的静态模型（`sft-LoRA-inference.ipynb`）

```python
# 将 merge 后的权重与 tokenizer 写到同一目录，后续可直接 AutoModel.from_pretrained(merged_dir)
merged_model.save_pretrained("/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/merged")
tokenizer.save_pretrained("/root/autodl-tmp/.autodl/sft/Qwen3-8B/sft-LoRA/merged")
```

---

### 11）关键路径与产物小结（配置级说明）

| 配置项 | 示例值 | 含义 |
|--------|--------|------|
| `HF_HOME` | `/root/autodl-tmp/hf` | Hub 缓存根目录 |
| `TENSORBOARD_LOGGING_DIR` | `/root/tf-logs` | TensorBoard 事件文件目录 |
| `output_dir` | `.../sft-LoRA` | 训练 checkpoint 根目录 |
| `save_model(...)` | `.../sft-LoRA/best` | 另存的适配器目录，供 `PeftModel.from_pretrained` |
| `merged`（可选） | `.../sft-LoRA/merged` | 合并后整模型 + tokenizer |

---

## 3.4.3 测试

1. **训练**  
   - 在 `full_para_FT` 目录下打开 **`sft-LoRA-train.ipynb`**。  
   - 执行首个单元完成 `pip install` 后 **重启内核**，再 **自上而下** 依次运行所有代码单元。  
   - 正常结束时，`output_dir` 下应有 checkpoint；执行 `save_model` 后应在 **`.../sft-LoRA/best`** 下看到适配器相关文件。  
   - 若开启 TensorBoard：在终端执行 `tensorboard --logdir /root/tf-logs`（远程需端口映射）。

2. **推理**  
   - 打开 **`sft-LoRA-inference.ipynb`**，将 `PeftModel.from_pretrained` 的路径与训练保存的 **`best`** 目录一致，依次运行至 `generate`。  
   - 若需单目录部署：运行 **「导出 merged」** 两格，之后可从 `merged` 目录直接加载整模型。

3. **验收**  
   - 训练 loss/eval 曲线或指标合理；  
   - 同分布样例下输出关键词格式与标注习惯一致；  
   - 显存不足时：减小 `per_device_train_batch_size`、设置 `max_length`、开启 `gradient_checkpointing` 等（在 `SFTConfig` 中按版本支持情况添加）。

---

**说明**：本文档章节编号 **3.4.x** 用于与课程/论文体例对齐；若全文仅含本实现，也可在目录中改称为「第三节」等，不影响内容与代码对应关系。
