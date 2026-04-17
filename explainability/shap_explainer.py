"""
SHAP 归因实现

模块功能：
    使用 shap 库对文本分类模型进行 token 级归因分析。
    通过 SHAP 值量化每个 token 对模型预测的边际贡献。

KernelSHAP 计算原理：
    SHAP 基于合作博弈论中的 Shapley 值，将每个 token 视为一个"参与者"，
    其 Shapley 值表示该 token 在所有可能的 token 子集组合中对模型输出的
    平均边际贡献。KernelSHAP 通过加权线性回归高效近似 Shapley 值，
    避免了枚举所有子集的指数级复杂度。

计算开销特点：
    尽管 KernelSHAP 相比精确 Shapley 值大幅降低了计算量，但对于
    BERT 类大模型，每次推理仍较耗时。因此本模块默认仅对测试集中
    随机采样的少量样本进行解释，采样数量由 config.SHAP_SAMPLE_SIZE 控制。

采样策略的权衡：
    较大的采样数量能更全面地反映模型的解释行为模式，但计算时间线性增长。
    较小的采样数量可能无法覆盖数据集中的多样化语义模式。实践中建议根据
    可用计算资源调整采样数量。

依赖模块：
    - models/bert_classifier.py：BertTextClassifier 模型
    - config.py：超参数配置

作者：Kris
"""

import torch
import numpy as np
import shap
from transformers import BertTokenizer, pipeline, BertForSequenceClassification

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, DEVICE, SHAP_SAMPLE_SIZE


def _create_prediction_function(model, tokenizer, max_seq_len=128):
    """
    创建供 SHAP 使用的预测函数包装器

    SHAP 的 Explainer 需要一个接受文本列表并返回预测概率数组的函数。
    此函数将 BertTextClassifier 包装为该接口。

    参数：
        model: BertTextClassifier 实例
        tokenizer: BertTokenizer 实例
        max_seq_len: 最大序列长度

    返回值：
        predict_fn: 接受文本列表，返回预测概率的函数
    """
    model.eval()

    def predict_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)

        encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_seq_len, return_tensors="pt",
            return_token_type_ids=True
        )

        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)
        token_type_ids = encodings["token_type_ids"].to(DEVICE)

        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return probs

    return predict_fn


def explain_batch(model, texts, tokenizer=None):
    """
    对一批文本进行 SHAP 归因分析

    参数：
        model: BertTextClassifier 实例
        texts: 文本字符串列表
        tokenizer: BertTokenizer 实例，若为 None 则自动加载

    返回值：
        list: 每条样本的归因结果字典列表，每个字典包含：
            - text: 原始文本
            - tokens: token 列表（按词粒度）
            - attribution_scores: 各 token 的 SHAP 值列表
            - predicted_label: 模型预测类别
            - true_label: None（由调用方填充）
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    predict_fn = _create_prediction_function(model, tokenizer)

    # 使用 shap.Explainer，masker 采用文本分词策略
    # shap 的文本 masker 会按词粒度进行遮蔽
    masker = shap.maskers.Text(tokenizer=r"\s+")
    explainer = shap.Explainer(predict_fn, masker, output_names=None)

    results = []

    for text in texts:
        try:
            # 计算 SHAP 值
            shap_values = explainer([text])

            # 获取模型预测
            probs = predict_fn(text)
            predicted_label = int(np.argmax(probs[0]))

            # 提取 token 列表和对应的 SHAP 值
            # shap_values.data 包含分词后的 token
            sample_tokens = list(shap_values.data[0])

            # shap_values.values 形状: (1, num_tokens, num_classes)
            # 取预测类别对应的 SHAP 值
            sample_scores = shap_values.values[0][:, predicted_label].tolist()

            results.append({
                "text": text,
                "tokens": sample_tokens,
                "attribution_scores": sample_scores,
                "predicted_label": predicted_label,
                "true_label": None,
            })
        except Exception as e:
            print(f"SHAP 解释失败: {e}")
            # 跳过失败的样本
            continue

    return results
