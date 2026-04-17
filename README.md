# 基于可解释性方法的 Transformer 模型可视化研究——以教育行为分析为例

作者：Kris

---

## 项目简介

本项目以教育行为分析和工程故障诊断为应用背景，基于预训练 BERT 模型完成文本分类任务，并在此基础上系统实现多种可解释性分析方法，包括注意力权重可视化、Integrated Gradients、SHAP、LIME 以及面向 Vision Transformer 的 Grad-CAM 适配。项目同时提供基于 Streamlit 的交互式可视化分析界面，供研究者在单条样本上观察各方法的归因结果并进行多方法对比。所有实验结果以 JSON 格式保存，供后续独立绘图脚本调用。

---

## 目录结构

```
ccf-shap/
├── main.py                        # 一键启动入口
├── requirements.txt               # 项目依赖
├── README.md                      # 本文件
├── config.py                      # 全局配置
├── data/                          # 数据加载与生成模块
├── models/                        # 模型定义
├── train/                         # 训练与评估
├── explainability/                # 可解释性方法实现
├── evaluation/                    # 忠实度与敏感度评估
├── results/                       # 结果保存工具
├── app/                           # Streamlit 界面
└── Results/                       # 实验结果输出目录
```

---

## 数据集

**教育场景：** SST-2（斯坦福情感树库二分类版本），通过 HuggingFace `datasets` 库自动下载，代表教育文本的情感倾向分类任务。

**工程场景：** CWRU 轴承故障数据集的文本描述形式，由项目内置脚本 `data/generate_cwru_text.py` 自动生成，包含正常状态、内圈故障、外圈故障、滚动体故障四个类别的文本描述。

---

## 可解释性方法

| 方法 | 类别 | 实现模块 |
|------|------|----------|
| 注意力权重可视化 | 注意力分析 | `explainability/attention_viz.py` |
| Integrated Gradients | 基于梯度的归因 | `explainability/integrated_gradients.py` |
| SHAP | 基于博弈论的归因 | `explainability/shap_explainer.py` |
| LIME | 基于局部代理的归因 | `explainability/lime_explainer.py` |
| Grad-CAM（ViT 适配） | 基于梯度的视觉归因 | `explainability/gradcam_vit.py` |

---

## 评估指标

**忠实度（Faithfulness）：** 通过逐步遮蔽高归因值 token 后观察模型预测概率的变化幅度来度量，以 AUC-Drop 指标量化，值越高表示归因结果与模型决策的吻合程度越好。

**敏感度（Sensitivity）：** 通过对输入文本施加同义词替换扰动，计算原始归因向量与扰动后归因向量之间的余弦相似度，均值越高表示归因结果的稳定性越好。

---

## 快速开始

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

## 结果输出

所有实验结果保存至 `D:\systemfiles\ccf-shap\Results\` 目录，文件列表如下：

| 文件名 | 内容描述 |
|--------|----------|
| `sst2_classification.json` | SST-2 分类性能指标及预测结果 |
| `cwru_classification.json` | CWRU 文本分类性能指标及预测结果 |
| `attention_sst2.json` | SST-2 测试样本的注意力权重归因结果 |
| `attention_cwru.json` | CWRU 测试样本的注意力权重归因结果 |
| `ig_sst2.json` | SST-2 的 Integrated Gradients 归因结果 |
| `ig_cwru.json` | CWRU 的 Integrated Gradients 归因结果 |
| `shap_sst2.json` | SST-2 的 SHAP 归因结果 |
| `shap_cwru.json` | CWRU 的 SHAP 归因结果 |
| `lime_sst2.json` | SST-2 的 LIME 归因结果 |
| `lime_cwru.json` | CWRU 的 LIME 归因结果 |
| `faithfulness_results.json` | 各方法的忠实度评估指标 |
| `sensitivity_results.json` | 各方法的敏感度评估指标 |

JSON 文件统一包含 `experiment_name`、`dataset`、`timestamp`、`author` 等元数据字段，字段结构详见各模块 docstring。

---

## 依赖环境

- Python 3.9+
- PyTorch 2.0+
- transformers
- datasets
- captum
- shap
- lime
- nltk
- scikit-learn
- streamlit

---

## 注意事项

- SHAP 方法计算开销较大，默认对测试集采样若干条样本进行解释，采样数量可在 `config.py` 的 `SHAP_SAMPLE_SIZE` 中调整。
- Grad-CAM 模块面向 ViT 图像分类模型，作为独立演示模块运行，与文本分类主流程相互独立，可单独调用。
- 首次运行 `train_sst2` 时将自动从 HuggingFace Hub 下载模型权重与数据集，需保持网络连接。
- 训练完成的模型权重保存于 `models/saved/` 目录，后续可解释性分析步骤将自动加载。
- `Results/` 目录在首次写入结果时自动创建，无需手动建立。

---

## 参考文献

本项目方法实现参考了以下工作：

- Yeh C, Chen Y, Wu A, et al. AttentionViz: A global view of transformer attention. IEEE TVCG, 2024.
- Ahmed M, et al. Integrated gradients-based defense against adversarial word substitution attacks. NCA, 2025.
- Salih A, et al. A perspective on explainable artificial intelligence methods: SHAP and LIME. Advanced Intelligent Systems, 2025.
- Choi H, Jin S, Han K. ICEv2: Interpretability, comprehensiveness, and explainability in vision transformer. IJCV, 2025.
- Mariotti E, et al. TextFocus: Assessing the faithfulness of feature attribution methods explanations in NLP. IEEE Access, 2024.
- Pawlicki M, et al. Evaluating the necessity of the multiple metrics for assessing explainable AI. Neurocomputing, 2024.
