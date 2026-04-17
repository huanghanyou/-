"""
LIME 局部线性代理模型归因实现

模块功能：
    使用 lime 库的 LimeTextExplainer 对文本分类模型进行局部解释。
    通过在输入文本周围构造扰动邻域，拟合局部线性模型来近似模型的决策边界，
    从而得到各词的归因权重。

扰动邻域的构造方式：
    LIME 通过随机遮蔽（删除）输入文本中的词语来构造扰动样本。
    每个扰动样本是原始文本的一个子集，被遮蔽的词语从文本中移除。
    LIME 在这些扰动样本上查询原始模型的预测概率，收集（扰动表示, 预测概率）
    数据对。

局部线性模型的拟合目标：
    LIME 在扰动样本的二值特征空间（每个词是否被保留）上拟合一个加权线性模型。
    权重由扰动样本与原始输入的相似度决定（核函数），距离越近的扰动样本权重越大。
    线性模型的系数即为各词的归因权重，正值表示正面贡献，负值表示负面贡献。

解释结果的稳定性特点：
    由于扰动过程具有随机性，同一输入的多次 LIME 解释可能产生不同结果。
    这种不稳定性是 LIME 方法的固有特性，可通过增加扰动样本数量（num_samples）
    来缓解，但无法完全消除。

依赖模块：
    - models/bert_classifier.py：BertTextClassifier 模型
    - config.py：超参数配置

作者：Kris
"""

import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_SEQ_LEN, DEVICE


def _create_prediction_function(model, tokenizer, max_seq_len=128):
    """
    创建供 LIME 使用的预测函数包装器

    LIME 需要一个接受文本列表并返回预测概率数组的函数。

    参数：
        model: BertTextClassifier 实例
        tokenizer: BertTokenizer 实例
        max_seq_len: 最大序列长度

    返回值：
        predict_fn: 接受文本列表，返回概率数组（shape: num_texts x num_classes）
    """
    model.eval()

    def predict_fn(texts):
        encodings = tokenizer(
            list(texts), padding="max_length", truncation=True,
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


def explain_sample(model, text, num_labels=2, num_samples=500, tokenizer=None):
    """
    对单条文本进行 LIME 归因分析

    参数：
        model: BertTextClassifier 实例
        text: 输入文本字符串
        num_labels: 分类类别数
        num_samples: LIME 扰动样本数量，越大结果越稳定但计算时间越长
        tokenizer: BertTokenizer 实例，若为 None 则自动加载

    返回值：
        dict: 包含以下字段
            - text: 原始文本
            - tokens: 词列表（按空格分词）
            - attribution_scores: 各词的 LIME 归因权重列表
            - predicted_label: 模型预测类别
            - true_label: None（由调用方填充）
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    predict_fn = _create_prediction_function(model, tokenizer, MAX_SEQ_LEN)

    # 创建 LIME 文本解释器
    # class_names 指定类别名称列表
    explainer = LimeTextExplainer(
        class_names=[str(i) for i in range(num_labels)]
    )

    # 获取模型预测类别
    probs = predict_fn([text])
    predicted_label = int(np.argmax(probs[0]))

    # 生成 LIME 解释
    # num_features 控制解释中包含的最大特征（词）数量
    # num_samples 控制用于拟合局部模型的扰动样本数量
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=len(text.split()),
        num_samples=num_samples,
        labels=(predicted_label,)
    )

    # 提取归因权重
    # explanation.as_list(label) 返回 [(word, weight), ...] 列表
    lime_weights = dict(explanation.as_list(label=predicted_label))

    # 按原始文本的词序整理归因结果
    words = text.split()
    attribution_scores = []
    for word in words:
        # LIME 可能对某些词未给出权重（如被完全遮蔽的情况），默认为 0
        score = lime_weights.get(word, 0.0)
        attribution_scores.append(float(score))

    return {
        "text": text,
        "tokens": words,
        "attribution_scores": attribution_scores,
        "predicted_label": predicted_label,
        "true_label": None,
    }
