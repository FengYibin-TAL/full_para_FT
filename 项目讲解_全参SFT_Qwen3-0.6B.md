# 基于 Qwen3-0.6B 的关键词抽取全参监督微调（SFT）项目讲解

本文档按「项目介绍 → 需求与目标 → 分步实现与配置」组织，代码与路径以仓库内 `sft-helloworld-train.ipynb`、`sft-helloworld-inference.ipynb` 为准。基座模型 **Qwen3-0.6B** 来自阿里通义千问（Qwen）开源系列；**全参微调**指训练时更新基座全部可训练权重（与 LoRA 等参数高效方法相对）。

---

## 一、项目介绍

### 1.1 项目是什么

本项目是在本地或云端 GPU 环境中，使用 **Hugging Face Transformers** 与 **TRL（Transformer Reinforcement Learning 库，现广泛用于 SFT）**，对小型因果语言模型 **Qwen/Qwen3-0.6B** 做 **监督微调（Supervised Fine-Tuning, SFT）**。任务场景为：**根据用户给出的长文本（含指令），让模型输出关键词串**（数据里多为分号分隔形式，如 `关键词1;关键词2`）。

实现形态以 **Jupyter Notebook** 为主：训练与推理拆成两个笔记本，数据为 JSON Lines（`.jsonl`），便于扩展与版本管理。

### 1.2 仓库内相关文件

| 文件/目录 | 作用 |
|-----------|------|
| `data/keywords_data_train.jsonl` | 训练对话数据 |
| `data/keywords_data_test.jsonl` | 验证/测试对话数据 |
| `sft-helloworld-train.ipynb` | 全参 SFT 全流程（环境、加载模型、数据、训练、保存） |
| `sft-helloworld-inference.ipynb` | 加载微调产物并做生成式推理 |
| `README.md` | 项目速览与路径说明 |

> 说明：与「图像去噪」等多 `.py` 分包结构不同，本示例将流程集中在 Notebook 中，下面「分步实现」按**执行顺序**讲解，每一部分对应 Notebook 中的连续单元格。

---

## 二、需求介绍

### 2.1 业务/任务需求

- **输入**：自然语言指令 + 待分析文本（如「关键词抽取：…」「抽取出文本中的关键词：…」等，与训练数据风格一致为佳）。
- **输出**：模型生成的 **关键词列表字符串**（格式需与标注习惯一致，便于下游解析）。
- **质量期望**：在领域文本上比未微调的基座模型更稳定地遵循「只输出关键词、少废话」等行为（具体取决于数据覆盖与训练量）。

### 2.2 技术需求

| 维度 | 要求说明 |
|------|----------|
| 算力 | GPU 推荐；0.6B 全参对显存要求远低于 7B/8B，但仍需按 batch 与序列长度权衡 |
| 依赖 | `transformers`、`trl`、`datasets`、`torch` 等；TensorBoard 可选 |
| 数据 | 对话式 JSONL，可映射为 TRL 所需的 `messages` 字段 |
| 可复现 | 固定随机种子（若需在脚本中补充）、记录 `output_dir` 与超参 |
| 存储 | 权重与缓存建议放在**数据盘**（如 AutoDL 扩容盘），避免系统盘占满 |

### 2.3 非目标（边界）

- 本文档**不**展开 LoRA/QLoRA（同目录另有 `sft-LoRA-*.ipynb` 可参考）。
- **不**涵盖模型上线、量化、推理服务框架（vLLM 等），仅到「训练 + Notebook 推理」。

---

## 三、分步实现与配置讲解

以下顺序与 `sft-helloworld-train.ipynb` 从上到下执行顺序一致；路径以示例机器为准，换环境时请统一替换为你的根目录。

---

### 第一步：环境与 Hugging Face 下载配置

**目的**：保证能稳定从 Hub 拉取 `Qwen/Qwen3-0.6B`，并把缓存放到大磁盘。

**典型配置**（Notebook 首个代码单元）：

```python
# ------------ Hugging Face 镜像与缓存目录 ------------
# HF_ENDPOINT：国内常用镜像，无法访问官方时可改为镜像站
%env HF_ENDPOINT=https://hf-mirror.com
# HF_HOME：模型与数据集缓存根目录，建议指向数据盘（需存在写权限）
%env HF_HOME=/root/autodl-tmp/hf

# 安装/更新训练所需库（版本随环境变化，以实际 pip 输出为准）
!pip install trl datasets
```

**讲解要点**：

- `HF_ENDPOINT`：解决「连不上 huggingface.co」的问题；若本机已能直连，可改回官方或删除该行按默认行为。
- `HF_HOME`：所有 `from_pretrained` 的缓存会进该目录，**与 `output_dir`（训练输出）不同**，勿混淆。
- `trl` 提供 `SFTTrainer`、`SFTConfig`；`datasets` 用于 `load_dataset('json', data_files=...)`。

---

### 第二步：加载基座模型与分词器（全参微调前提）

**目的**：把 **Qwen3-0.6B** 与对应 **Tokenizer** 载入内存/显存，后续 SFT 将更新**全部**参与训练的权重。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 基座模型 ID（Hub）或本地路径；此处为通义千问 0.6B 小规模实验用模型
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**讲解要点**：

- `AutoModelForCausalLM`：因果 LM，适合「给定上文预测下文」的聊天/补全形式。
- **全参微调**不在此包 LoRA；所有 `requires_grad=True` 的参数都会参与反向传播（具体还受 Trainer 冻结策略影响，本示例未额外冻结）。
- 首次下载体积与磁盘占用需预留；重复运行会读缓存。

---

### 第三步：数据加载与格式转换（对接 TRL）

**目的**：把业务 JSONL 转成 TRL `SFTTrainer` 能吃的 **`messages`** 列表（OpenAI 风格 `role` + `content`）。

数据每条大致结构：顶层含 `conversation` 数组，元素为 `human` / `assistant` 字符串对（可多轮）。

```python
from datasets import load_dataset

# 同时注册训练集、测试集文件路径（相对于 Notebook 工作目录，一般在 full_para_FT 下执行）
dataset_dict = load_dataset(
    "json",
    data_files={
        "train": "data/keywords_data_train.jsonl",
        "test": "data/keywords_data_test.jsonl",
    },
)

def map_func(example):
    """将单条样本中的多轮 human/assistant 转为 messages 列表。"""
    conversation = example["conversation"]
    messages = []
    for item in conversation:
        messages.append({"role": "user", "content": item["human"]})
        messages.append({"role": "assistant", "content": item["assistant"]})
    return {"messages": messages}

# batched=False：逐条 map，逻辑简单；去掉与训练无关的列减少存储
dataset_dict = dataset_dict.map(
    map_func,
    batched=False,
    remove_columns=["dataset", "conversation", "category", "conversation_id"],
)
```

**讲解要点**：

- `load_dataset('json', data_files=...)`：**不要求**先把 JSONL 转成 Hugging Face Hub 上的数据集，适合本地实验。
- `messages` 格式是 TRL SFT 管道与 Qwen 聊天模板的常见输入形式。
- `remove_columns`：避免多余字段干扰且略减内存；**不要删掉** `messages`。

**建议自检**：执行 `dataset_dict["train"][0]`，确认样本中只有合法字段且 `messages` 内容符合预期。

---

### 第四步：训练超参与 `SFTConfig`

**目的**：指定**输出目录、步数、batch、学习率、评估与保存策略**，并**打开 TensorBoard**（Transformers 5.x 约定）。

```python
import os
from trl import SFTConfig, SFTTrainer

# ------------ TensorBoard 日志目录（Transformers 5.x）------------
# 必须在创建 SFTTrainer 之前设置；勿仅依赖已废弃的 TrainingArguments.logging_dir
os.environ["TENSORBOARD_LOGGING_DIR"] = "/root/tf-logs"

training_args = SFTConfig(
    # 检查点与日志根目录（建议数据盘）
    output_dir="/root/autodl-tmp/.autodl/sft/Qwen3-0.6B/sft-full",
    max_steps=1000,                      # 最大优化步数（与 epoch 二选一逻辑由策略决定，此处按步数训练）
    per_device_train_batch_size=4,       # 单卡 batch，显存不足可调小
    learning_rate=5e-5,                  # 全参微调常用量级之一，需与数据规模联调
    logging_steps=10,                    # 每 N 步打印/记录日志
    report_to="tensorboard",             # 关键：默认 "none" 不写 tfevents，必须显式打开
    save_steps=100,                      # 每 N 步保存检查点
    save_total_limit=2,                  # 最多保留检查点份数，节省磁盘
    eval_strategy="steps",               # 按步评估
    eval_steps=100,
    load_best_model_at_end=True,         # 训练结束加载验证最优权重（依赖保存与评估）
    bf16=True,                           # 半精度，需 GPU 支持 bf16
    warmup_steps=50,                     # 学习率预热步数
)
```

**讲解要点**：

- **`report_to="tensorboard"`**：否则不会挂载 `TensorBoardCallback`，`/root/tf-logs` 下不会出现有效 `tfevents`。
- **`TENSORBOARD_LOGGING_DIR`**：当前版本 TensorBoard 回调读的是该环境变量；与 `output_dir` 不同。
- **`bf16=True`**：若硬件不支持，可改为 `fp16=True` 或关闭混合精度（显存占用上升）。
- **`output_dir`**：全量权重、optimizer 状态、scheduler 等与 Trainer 相关的文件默认写在此树下。

---

### 第五步：构造 `SFTTrainer` 并理解数据管线的输入

**目的**：将 **模型、参数、训练/评估集、分词器（processing_class）** 绑定到统一训练循环。

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
)
```

**讲解要点**：

- `processing_class=tokenizer`：TRL 会用分词器将 `messages` 编成模型输入（具体模板与 Qwen3 的 chat template 一致，由 tokenizer 配置决定）。
- **可选自检**：`trainer.get_train_dataloader()` 取一个 batch，`tokenizer.decode(batch['input_ids'][0])` 查看拼接后的单条样本是否符合对话格式。

---

### 第六步：训练、保存最佳模型

**启动训练**：

```python
trainer.train()
```

**另存为固定推理路径（便于推理 Notebook 引用）**：

```python
trainer.save_model("/root/autodl-tmp/.autodl/sft/Qwen3-0.6B/sft-full/best")
```

**讲解要点**：

- `train()` 会按 `SFTConfig` 做日志、评估、保存；若中断需考虑从 checkpoint 恢复（本示例未展开）。
- `save_model(...)` 写出适配 `from_pretrained` 的目录；推理时把 `model_name` 指到该目录即可。

**TensorBoard**：在产生日志后于终端执行 `tensorboard --logdir /root/tf-logs`（远程需端口映射）。可用 `find /root/tf-logs -name '*tfevents*'` 确认是否已有事件文件。

---

### 第七步：推理验证（`sft-helloworld-inference.ipynb`）

**目的**：加载 **微调后的权重目录**，用 **聊天模板** 组 prompt，调用 `generate` 得到关键词结果。

**环境与路径**：与训练类似设置 `HF_ENDPOINT` / `HF_HOME`；`model_name` 指向保存的 `best` 目录。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/root/autodl-tmp/.autodl/sft/Qwen3-0.6B/sft-full/best"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
)
```

**组消息与模板**（Qwen3 支持思考模式开关，此处示例关闭以简化输出）：

```python
prompt = "抽取出文本中的关键词：\n标题：……\n文本：……"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

**生成与解码**（与 `sft-helloworld-inference.ipynb` 一致；其中对 `output_ids` 按特殊 token 切分「思考 / 正文」需与当前 Qwen3 tokenizer 配置一致）：

```python
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
# 随后用 notebook 中的 rindex(151668) 等方式划分 thinking_content 与 content，再 decode
```

**讲解要点**：

- 推理务必 **`eval` 模式**（若手写循环需 `model.eval()`）；Notebook 中 `generate` 一般已合理。
- `max_new_tokens=32768` 为仓库当前写法；**关键词抽取只需短输出**，实际使用建议改为几十到数百，以省显存与时间并降低跑飞概率。
- 若加载报错，先检查 `best` 目录是否包含 `config.json`、权重分片等完整文件。

---

## 四、测试与验收建议

1. **数据**：训练/测试条数与 `dataset_dict` 键名正确；随机抽几条人工读 `messages`。
2. **训练**：`loss` 随步数下降（允许波动）；验证集指标在可接受范围。
3. **磁盘**：`output_dir` 下出现 checkpoint；`best` 目录可 `from_pretrained`。
4. **TensorBoard**：`report_to` 与 `TENSORBOARD_LOGGING_DIR` 已设，且 `tfevents` 非空。
5. **推理**：同分布样例输出格式与训练标注风格一致；换领域样例观察鲁棒性。

---

## 五、常见问题（简要）

| 现象 | 可能原因 |
|------|----------|
| TensorBoard 无曲线 | 未设 `report_to="tensorboard"`；或 log 目录无 `tfevents`；或 `--logdir` 指错 |
| 显存 OOM | 减小 `per_device_train_batch_size` 或序列长度相关配置（若在 Config 中启用） |
| 推理胡言乱语 | 步数不足、学习率过大、或 prompt 与训练分布差太远 |
| 下载失败 | 镜像不可达或 `HF_HOME` 无写权限 |

---

## 六、小结

本项目从**需求**上完成「关键词抽取」任务的 **Qwen3-0.6B 全参 SFT**；从**实现**上按顺序完成 **Hub 配置 → 模型与分词器 → JSONL→messages → SFTConfig（含 TensorBoard）→ SFTTrainer → train/save → 推理 Notebook 验证**。按本文档各步检查配置与路径，即可在同类环境中复现或改写为多文件工程结构。
