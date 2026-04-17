# Transformer 模型可解释性分析与可视化

基于可解释性方法的 Transformer 模型可视化研究——以教育行为分析与工程故障诊断为例

**语言：** Python 3.9+  
**框架：** PyTorch 2.0+

---

## 📋 项目简介

本项目实现了一套完整的 **文本分类 + 可解释性分析 + 评估 + 可视化** 工作流。项目包含：

- **文本分类模型**：基于预训练 BERT 模型，在 SST-2（情感分析）和 CWRU（故障诊断）两个数据集上进行文本分类
- **多种可解释性方法**：注意力权重可视化、Integrated Gradients、SHAP、LIME、Grad-CAM（ViT 适配）
- **评估指标**：忠实度（Faithfulness）和敏感度（Sensitivity）评估
- **交互式界面**：基于 Streamlit 的 Web 可视化分析工具，支持单样本多方法对比
- **完整输出**：所有实验结果以 JSON 格式保存，便于后续分析和绘图

---

## 📁 项目结构

```
.
├── main.py                        # 项目入口：命令行参数分发与各模式协调
├── config.py                      # 全局配置：路径、超参数、设备配置
├── requirements.txt               # Python 依赖列表
├── README.md                      # 项目文档（本文件）
│
├── data/                          # 数据加载与处理模块
│   ├── dataset_loader.py          # SST-2 数据集加载（HuggingFace datasets）
│   ├── cwru_text_dataset.py       # CWRU 文本数据集加载与生成
│   └── __init__.py
│
├── models/                        # 模型定义
│   ├── bert_classifier.py         # BERT 文本分类模型（2分类/4分类）
│   ├── saved/                     # 训练好的模型权重保存目录
│   └── __init__.py
│
├── train/                         # 训练与评估模块
│   ├── trainer.py                 # 训练循环、验证、模型加载/保存
│   ├── evaluator.py               # 分类性能评估（准确率、精确率、召回率、F1）
│   └── __init__.py
│
├── explainability/                # 可解释性方法实现
│   ├── attention_viz.py           # 注意力权重可视化
│   ├── integrated_gradients.py    # Integrated Gradients 梯度归因
│   ├── shap_explainer.py          # SHAP (SHapley Additive exPlanations)
│   ├── shap_explainer_optimized.py # SHAP 优化版本
│   ├── lime_explainer.py          # LIME (Local Interpretable Model-agnostic)
│   ├── gradcam_bert.py            # Grad-CAM 适配 BERT
│   ├── gradcam_vit.py             # Grad-CAM 适配 Vision Transformer（演示）
│   ├── attribution_unified.py     # 统一的归因接口（实验模块）
│   └── __init__.py
│
├── evaluation/                    # 忠实度与敏感度评估
│   ├── faithfulness.py            # 忠实度评估（AUC-Drop）
│   ├── faithfulness_advanced.py   # 高级忠实度评估
│   ├── sensitivity.py             # 敏感度评估（扰动稳定性）
│   ├── robustness_evaluation.py   # 稳健性评估（实验模块）
│   └── __init__.py
│
├── results/                       # 结果保存工具
│   ├── result_saver.py            # JSON 结果保存与加载接口
│   └── __init__.py
│
├── app/                           # Streamlit Web 界面
│   ├── streamlit_app.py           # 交互式可视化分析应用
│   └── __init__.py
│
├── visualization.py               # 可视化工具函数（独立脚本）
├── test_methods.py                # 单元测试和方法验证
│
├── Results/                       # 实验结果输出目录（自动创建）
│   ├── sst2_classification.json   # SST-2 分类指标
│   ├── cwru_classification.json   # CWRU 分类指标
│   ├── attention_sst2.json        # SST-2 注意力归因
│   ├── attention_cwru.json        # CWRU 注意力归因
│   ├── ig_sst2.json               # SST-2 IG 归因
│   ├── ig_cwru.json               # CWRU IG 归因
│   ├── shap_sst2.json             # SST-2 SHAP 归因
│   ├── shap_cwru.json             # CWRU SHAP 归因
│   ├── lime_sst2.json             # SST-2 LIME 归因
│   ├── lime_cwru.json             # CWRU LIME 归因
│   ├── gradcam_sst2.json          # SST-2 Grad-CAM 归因
│   ├── gradcam_cwru.json          # CWRU Grad-CAM 归因
│   ├── faithfulness_results.json  # 忠实度评估指标
│   └── sensitivity_results.json   # 敏感度评估指标
│
└── 中期报告/                       # 中期报告文档
```

---

## 📊 数据集

### SST-2（Stanford Sentiment Treebank 2）

- **来源**：HuggingFace `datasets` 库自动下载
- **任务**：电影评论二分类（正面/负面情感）
- **规模**：训练集 67K，验证集 872，测试集 1821
- **应用背景**：教育文本的情感倾向分析

### CWRU（Case Western Reserve University）

- **来源**：轴承故障数据集的文本形式
- **任务**：故障诊断四分类
- **类别**：
  - 正常状态（Normal）
  - 内圈故障（Inner Race Fault）
  - 外圈故障（Outer Race Fault）
  - 滚动体故障（Ball Fault）
- **生成方式**：由 `data/cwru_text_dataset.py` 自动生成文本描述
- **应用背景**：工程故障诊断

---

## 🔍 可解释性方法

本项目实现了 5 种可解释性方法，用于分析模型决策过程：

| 方法名称 | 类型 | 说明 | 模块 |
|---------|------|------|------|
| **注意力可视化** | 注意力分析 | 提取 BERT 各层的注意力权重，可视化 token 间的关系 | `attention_viz.py` |
| **Integrated Gradients** | 梯度归因 | 沿着从零向量到输入的路径积分梯度，获得特征重要性 | `integrated_gradients.py` |
| **SHAP** | 博弈论归因 | 基于 Shapley 值的模型解释方法，考虑特征的联合贡献 | `shap_explainer.py` |
| **LIME** | 局部代理 | 用局部可解释模型逼近黑盒模型，解释单个预测 | `lime_explainer.py` |
| **Grad-CAM** | 梯度可视化 | 原本用于 CNN，已适配 BERT 和 ViT（可选演示） | `gradcam_bert.py`, `gradcam_vit.py` |

### 方法对比

- **基于梯度**：注意力可视化、Integrated Gradients、Grad-CAM
  - 优点：计算高效，直接利用反向传播
  - 缺点：对梯度饱和问题敏感
  
- **基于扰动**：LIME、SHAP
   - 优点：模型无关，不依赖梯度信息
   - 缺点：计算开销大，需要多次前向传播

---

## 📈 评估指标

### 忠实度（Faithfulness）

**度量方法**：通过逐步遮蔽高归因值的 token，观察模型预测概率的变化幅度

**关键指标**：**AUC-Drop**
- 遮蔽重要 token 时，预测概率下降的面积
- 值越高，说明归因方法越能正确识别决策关键词
- 评估范围：BERT 的注意力、IG、SHAP、LIME、Grad-CAM

**配置**：
```python
MASKING_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

### 敏感度（Sensitivity）

**度量方法**：对输入文本施加同义词替换扰动，计算原始归因与扰动后归因的一致性

**关键指标**：**余弦相似度**
- 衡量扰动前后归因结果的稳定性
- 值越高，表示模型解释越稳健，不易被文本表述变化所影响
- 评估方法：注意力、IG、LIME

**配置**：
```python
SENSITIVITY_PERTURB_RATIO = 0.2      # 替换 20% 的词语
SENSITIVITY_NUM_PERTURBATIONS = 5    # 每个样本生成 5 个扰动版本
```

---

## 🚀 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

初次运行前需下载 NLTK 数据：

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 运行方式

```bash
# 训练 SST-2 分类模型
python main.py --mode train_sst2

# 训练 CWRU 文本分类模型
python main.py --mode train_cwru

# 对 SST-2 运行全部可解释性方法
python main.py --mode explain_sst2

# 对 CWRU 运行全部可解释性方法
python main.py --mode explain_cwru

# 运行忠实度与敏感度评估
python main.py --mode evaluate_explainability

# 启动 Streamlit 可视化界面
python main.py --mode app

# 依次执行所有步骤
python main.py --mode all
```

---

## 💾 结果输出

所有实验结果保存至 `Results/` 目录（相对于项目根目录），文件列表如下：

### 分类任务结果

| 文件名 | 内容描述 |
|--------|----------|
| `sst2_classification.json` | SST-2 分类性能指标及预测结果 |
| `cwru_classification.json` | CWRU 文本分类性能指标及预测结果 |

### 可解释性分析结果

| 文件名 | 内容描述 |
|--------|----------|
| `attention_sst2.json` | SST-2 样本的注意力权重归因（各层各头） |
| `attention_cwru.json` | CWRU 样本的注意力权重归因 |
| `ig_sst2.json` | SST-2 的 Integrated Gradients 归因结果 |
| `ig_cwru.json` | CWRU 的 Integrated Gradients 归因结果 |
| `shap_sst2.json` | SST-2 的 SHAP 归因结果（采样） |
| `shap_cwru.json` | CWRU 的 SHAP 归因结果（采样） |
| `lime_sst2.json` | SST-2 的 LIME 归因结果 |
| `lime_cwru.json` | CWRU 的 LIME 归因结果 |
| `gradcam_sst2.json` | SST-2 的 Grad-CAM 归因结果 |
| `gradcam_cwru.json` | CWRU 的 Grad-CAM 归因结果 |

### 评估结果

| 文件名 | 内容描述 |
|--------|----------|
| `faithfulness_results.json` | 各方法的忠实度评估（AUC-Drop）及曲线 |
| `sensitivity_results.json` | 各方法的敏感度评估（余弦相似度）及样本级分数 |

**数据格式**：所有 JSON 文件均包含以下元数据字段：
- `experiment_name`：实验类型名称
- `dataset`：数据集名称（sst2 或 cwru）
- `timestamp`：执行时间戳
- `author`：实验者信息

详细的数据结构说明见各模块的 docstring。

---

## 📦 依赖环境

### Python 版本
- **Python 3.9+**

### 核心依赖
```
PyTorch >= 2.0.0              # 深度学习框架
transformers >= 4.30.0        # HuggingFace Transformer 模型库
datasets >= 2.10.0            # HuggingFace 数据集库
```

### 可解释性方法依赖
```
captum >= 0.6.0               # Integrated Gradients、Grad-CAM
shap >= 0.41.0                # SHAP 解释方法
lime >= 0.2.0                 # LIME 解释方法
```

### 工具与可视化
```
scikit-learn >= 1.2.0         # 机器学习工具（评估指标）
nltk >= 3.8                   # 自然语言处理（同义词替换）
numpy >= 1.21.0               # 数值计算
pandas >= 1.3.0               # 数据处理
tqdm >= 4.60.0                # 进度条
streamlit >= 1.20.0           # Web 界面框架
plotly >= 5.10.0              # 交互式绘图
```

详见 `requirements.txt`。

---

## ⚙️ 关键配置参数

所有配置项在 `config.py` 中定义，可根据需求调整：

### 模型与数据配置
```python
MODEL_NAME = "bert-base-uncased"          # 预训练模型
MAX_SEQ_LEN = 128                         # 最大序列长度
BATCH_SIZE = 16                           # 训练批大小
```

### 训练超参数
```python
LEARNING_RATE = 2e-5                      # AdamW 学习率
NUM_EPOCHS = 3                            # 训练轮数
SEED = 42                                 # 随机种子
WARMUP_RATIO = 0.1                        # Warmup 比例
```

### 可解释性超参数
```python
EXPLAIN_SAMPLE_SIZE = 50                  # 可解释性分析采样数
SHAP_SAMPLE_SIZE = 20                     # SHAP 采样数（计算开销大）
IG_N_STEPS = 50                           # IG 积分步数
ATTENTION_TOP_K = 10                      # 注意力 Top-K 显示
```

### 评估超参数
```python
MASKING_RATIOS = [0.1, 0.2, ..., 1.0]    # 忠实度遮蔽比例
SENSITIVITY_PERTURB_RATIO = 0.2           # 敏感度扰动比例
SENSITIVITY_NUM_PERTURBATIONS = 5         # 每个样本的扰动数量
```

---

## 📝 使用说明

### 1. 环境配置

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 下载 NLTK 数据（用于敏感度评估中的同义词替换）
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. 运行模式

**单个任务运行**：

```bash
# 训练模型
python main.py --mode train_sst2          # 训练 SST-2 模型
python main.py --mode train_cwru          # 训练 CWRU 模型

# 可解释性分析
python main.py --mode explain_sst2        # SST-2 全部方法分析
python main.py --mode explain_cwru        # CWRU 全部方法分析

# 评估
python main.py --mode evaluate_explainability  # 忠实度 + 敏感度评估

# 可视化
python main.py --mode app                 # 启动 Streamlit Web 界面
```

**一键运行所有步骤**：

```bash
python main.py --mode all
```

流程：训练 SST-2 → 训练 CWRU → 分析 SST-2 → 分析 CWRU → 评估 → 完成

### 3. Streamlit 可视化界面

启动后，在浏览器访问 `http://localhost:8501`，可以：
- 查看分类性能指标
- 对比不同可解释性方法的归因结果
- 调整样本，观察模型决策依据
- 下载分析图表

---

## ⚠️ 注意事项

1. **网络连接**
   - 首次运行 `train_sst2` 会从 HuggingFace Hub 下载 BERT 模型和数据集（约 500MB）
   - 需保持网络连接畅通

2. **计算资源**
   - 推荐使用 GPU（NVIDIA CUDA）加速训练和分析
   - SHAP 方法计算开销较大，已通过采样（默认 20 个样本）降低开销
   - 可在 `config.py` 的 `SHAP_SAMPLE_SIZE` 调整采样数量

3. **磁盘空间**
   - 模型权重保存在 `models/saved/` 目录
   - 实验结果保存在 `Results/` 目录
   - 建议预留至少 2GB 空间

4. **模块依赖**
   - `models/saved/` 中必须有对应数据集的训练好的模型，才能运行可解释性分析和评估
   - Grad-CAM ViT 模块（`gradcam_vit.py`）为独立演示，不影响主流程

5. **随机性控制**
   - 所有随机操作使用 `SEED = 42` 保证可重复性
   - SHAP 和 LIME 的采样过程存在随机性，多次运行结果可能有微小差异

---

## 📚 项目亮点

✨ **完整的可解释性研究工作流**
- 从模型训练、多方法分析到性能评估的全链路实现

✨ **多数据集支持**
- 教育场景（SST-2 情感分析）和工程场景（CWRU 故障诊断）

✨ **全面的评估指标**
- 忠实度（AUC-Drop）+ 敏感度（余弦相似度）双重评估

✨ **交互式可视化**
- Streamlit Web 界面，支持单样本多方法对比

✨ **易于扩展**
- 模块化设计，易于添加新的可解释性方法或数据集

---

## 🔗 参考资料

本项目的实现参考了以下学术工作：

1. Yeh C, Chen Y, Wu A, et al. **AttentionViz: A global view of transformer attention**. IEEE Transactions on Visualization and Computer Graphics, 2024.

2. Ahmed M, et al. **Integrated gradients-based defense against adversarial word substitution attacks**. Neural Computing and Applications, 2025.

3. Salih A, et al. **A perspective on explainable artificial intelligence methods: SHAP and LIME**. Advanced Intelligent Systems, 2025.

4. Choi H, Jin S, Han K. **ICEv2: Interpretability, comprehensiveness, and explainability in vision transformer**. International Journal of Computer Vision, 2025.

5. Mariotti E, et al. **TextFocus: Assessing the faithfulness of feature attribution methods explanations in NLP**. IEEE Access, 2024.

6. Pawlicki M, et al. **Evaluating the necessity of the multiple metrics for assessing explainable AI**. Neurocomputing, 2024.

---

## 📞 联系与反馈

- **作者**：Kris
- **最后更新**：2026年4月18日
- **项目状态**：毕业论文项目

如有问题或建议，欢迎通过 Issue 或 Email 联系。
