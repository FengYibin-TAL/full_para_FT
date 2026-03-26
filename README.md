# 关键词抽取 · 全参 / LoRA 监督微调（SFT）

基于 **Hugging Face Transformers** 与 **TRL** 的入门示例：在对话式数据上对 **Qwen** 系列模型做 SFT，任务为从给定文本中抽取关键词（分号分隔等形式）。

## 项目结构

| 路径 | 说明 |
|------|------|
| `data/keywords_data_train.jsonl` | 训练集（约 4.95 万条对话样本） |
| `data/keywords_data_test.jsonl` | 验证/测试集 |
| `sft-helloworld-train.ipynb` | **全参微调**：`Qwen/Qwen3-0.6B` + `SFTTrainer` |
| `sft-helloworld-inference.ipynb` | 加载全参微调产物并推理 |
| `sft-LoRA-train.ipynb` | **LoRA 微调**：`Qwen/Qwen3-8B` + PEFT |
| `sft-LoRA-inference.ipynb` | LoRA 微调产物推理 |

## 数据格式

每行为一个 JSON 对象，核心字段为 `conversation`：多轮 `human` / `assistant` 字符串。Notebook 中会 `map` 为 TRL 所需的 `messages`（`role` + `content`），并去掉 `dataset`、`category`、`conversation_id` 等列。

## 环境依赖

- Python 3.10+（示例环境为 3.12）
- PyTorch（需与 CUDA 版本匹配）
- 主要 Python 包：`transformers`、`trl`、`datasets`、`accelerate`；LoRA 另需 `peft`
- 可选：`tensorboard`（用于训练曲线）

安装示例（在 Notebook 中已部分使用）：

```bash
pip install trl datasets accelerate peft tensorboard
```

## 使用说明

### 1. Hugging Face 与缓存（国内/镜像）

Notebook 中默认设置示例：

- `HF_ENDPOINT=https://hf-mirror.com`（镜像，可按需改为官方或自建）
- `HF_HOME` 指向数据盘上的缓存目录（如 AutoDL 扩容盘），避免占满系统盘

请按本机路径修改上述变量。

### 2. 全参微调（Hello World）

打开 `sft-helloworld-train.ipynb`，自上而下执行：

1. 安装依赖并配置环境变量  
2. 加载 `Qwen/Qwen3-0.6B` 与分词器  
3. 从 `data/*.jsonl` 构建数据集并格式化  
4. `SFTConfig` + `SFTTrainer` 训练  

**默认输出目录**（可在 Notebook 中改）：

- 检查点与最终权重：`/root/autodl-tmp/.autodl/sft/Qwen3-0.6B/sft-full/`  
- 脚本中会将最优模型另存为：`.../sft-full/best`（与推理 Notebook 一致）

**TensorBoard（Transformers 5.x）**

- 已启用 `report_to="tensorboard"`，并通过环境变量 `TENSORBOARD_LOGGING_DIR=/root/tf-logs` 指定日志目录（勿仅依赖已废弃的 `logging_dir` 参数）。  
- 训练产生日志后：`tensorboard --logdir /root/tf-logs`  
- 远程机器需做端口映射才能在本地浏览器访问。

### 3. 全参推理

打开 `sft-helloworld-inference.ipynb`，将 `model_name` 指向微调后的目录（默认 `.../sft-full/best`），执行加载与生成单元。

### 4. LoRA 微调

打开 `sft-LoRA-train.ipynb`：在 `Qwen3-8B` 上配置 `LoraConfig` 与 `SFTTrainer`。默认 `output_dir` 为 `/root/autodl-tmp/sft/Qwen3-8B/sft-LoRA`（请按显卡显存与磁盘调整）。

> 若需与全参 Notebook 一致的 TensorBoard 行为，请同样设置 `report_to="tensorboard"` 与 `TENSORBOARD_LOGGING_DIR`，勿单独依赖 `logging_dir`。

## 注意事项

- **路径**：文中 `/root/autodl-tmp/...` 为示例；克隆到其他机器时请全局替换为你的数据盘与输出路径。  
- **显存**：`0.6B` 全参与 `8B` LoRA 对显存要求不同，请按 GPU 调整 `per_device_train_batch_size`、`max_length` 等。  
- **Cursor / VS Code 打开 Notebook**：若历史单元格含 Jupyter 控件输出，可能出现 ipywidgets 渲染告警，不影响在 Jupyter 内核中重新运行；可清除输出后保存以减轻干扰。

## 许可证与第三方模型

训练数据与基座模型（如 Qwen）各自遵循其官方许可证；使用本仓库代码时请同时遵守数据与模型的条款。
