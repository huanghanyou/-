"""
忠实度评估模块

模块功能：
    通过基于特征遮蔽的方式评估各可解释性方法的忠实度。
    逐步遮蔽高归因值的 token 后观察模型预测概率的变化幅度，
    以 AUC-Drop 指标量化归因结果与模型决策的吻合程度。

Sufficiency 与 Comprehensiveness 的含义差异：
    - Comprehensiveness（全面性）：遮蔽高归因值 token 后，模型对原预测类别的
      概率下降幅度。下降越大，说明归因方法识别出的重要 token 确实对模型决策
      有关键作用，忠实度越高。
    - Sufficiency（充分性）：仅保留高归因值 token、遮蔽其余 token 后，
      模型对原预测类别的概率保持程度。保持越高，说明归因方法识别出的 token
      足以支撑模型决策。

遮蔽策略：
    将目标 token 替换为 [MASK] token。这种策略保持了序列长度不变，
    但可能对教育文本的语法结构造成破坏（如破坏主谓搭配），从而引入
    与归因无关的预测变化。这是该评估方法的固有局限。

依赖模块：
    - models/bert_classifier.py：BertTextClassifier 模型
    - explainability/ 各归因模块
    - config.py：超参数配置

作者：Kris
"""

import torch
import numpy as np
from transformers import BertTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_SEQ_LEN, DEVICE, MASKING_RATIOS


def _get_prediction_prob(model, input_ids, attention_mask, token_type_ids, target_class):
    """
    获取模型对指定类别的预测概率

    参数：
        model: BertTextClassifier 实例
        input_ids: token 编码序列
        attention_mask: 注意力掩码
        token_type_ids: 段落类型编码
        target_class: 目标类别索引

    返回值：
        float: 模型对目标类别的预测概率
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            input_ids.to(DEVICE),
            attention_mask.to(DEVICE),
            token_type_ids.to(DEVICE)
        )
        probs = torch.softmax(logits, dim=-1)
    return probs[0, target_class].item()


def compute_comprehensiveness(model, text, attribution_scores, tokenizer=None):
    """
    计算单条样本的 Comprehensiveness 指标

    遮蔽策略：按归因分数从高到低排序 token，逐步增加遮蔽比例，
    记录每个比例下模型预测概率的下降幅度。

    参数：
        model: BertTextClassifier 实例
        text: 原始文本
        attribution_scores: token 级归因分数列表
        tokenizer: BertTokenizer 实例

    返回值：
        dict: 包含以下字段
            - drop_curve: 各遮蔽比例下的概率下降值列表
            - auc_drop: 概率下降曲线下面积（越大表示忠实度越高）
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 获取 [MASK] token 的 id
    mask_token_id = tokenizer.mask_token_id

    # 分词
    encoding = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_SEQ_LEN, return_tensors="pt",
        return_token_type_ids=True
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]

    # 获取原始预测
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            input_ids.to(DEVICE),
            attention_mask.to(DEVICE),
            token_type_ids.to(DEVICE)
        )
        original_probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(original_probs, dim=1).item()
        original_prob = original_probs[0, predicted_class].item()

    # 计算有效 token 数量（排除 [CLS]、[SEP]、[PAD]）
    valid_len = attention_mask[0].sum().item()
    # 归因分数仅对应有效 token，排除 [CLS] 和 [SEP]
    num_attributable = min(len(attribution_scores), valid_len - 2)
    if num_attributable <= 0:
        return {"drop_curve": [0.0] * len(MASKING_RATIOS), "auc_drop": 0.0}

    # 按归因分数降序排列 token 索引（在 input_ids 中的位置，从 1 开始跳过 [CLS]）
    sorted_indices = np.argsort(attribution_scores[:num_attributable])[::-1]
    # 映射到 input_ids 中的实际位置（+1 跳过 [CLS]）
    sorted_positions = [int(idx) + 1 for idx in sorted_indices]

    drop_curve = []

    for ratio in MASKING_RATIOS:
        # 计算本轮需遮蔽的 token 数量
        num_mask = max(1, int(num_attributable * ratio))
        positions_to_mask = sorted_positions[:num_mask]

        # 复制 input_ids 并遮蔽指定位置
        masked_input_ids = input_ids.clone()
        for pos in positions_to_mask:
            if pos < MAX_SEQ_LEN:
                masked_input_ids[0, pos] = mask_token_id

        # 计算遮蔽后的预测概率
        masked_prob = _get_prediction_prob(
            model, masked_input_ids, attention_mask, token_type_ids, predicted_class
        )

        # 概率下降幅度
        drop = original_prob - masked_prob
        drop_curve.append(float(drop))

    # AUC-Drop: 概率下降曲线下面积（使用梯形法则近似）
    auc_drop = float(np.trapz(drop_curve, MASKING_RATIOS))

    return {
        "drop_curve": drop_curve,
        "auc_drop": auc_drop,
    }


def evaluate_faithfulness(model, texts, attribution_results, method_name,
                          dataset_name, tokenizer=None):
    """
    对一组样本评估某可解释性方法的忠实度

    参数：
        model: BertTextClassifier 实例
        texts: 文本列表
        attribution_results: 归因结果列表，每个元素包含 attribution_scores 字段
        method_name: 方法名称（用于结果标记）
        dataset_name: 数据集名称
        tokenizer: BertTokenizer 实例

    返回值：
        dict: 包含以下字段
            - method: 方法名称
            - dataset: 数据集名称
            - mean_auc_drop: 平均 AUC-Drop 分数
            - auc_drop_curves: 各遮蔽比例下的平均概率下降值
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    all_auc_drops = []
    all_curves = []

    for i, (text, attr_result) in enumerate(zip(texts, attribution_results)):
        scores = attr_result.get("attribution_scores", [])
        if not scores:
            continue

        result = compute_comprehensiveness(model, text, scores, tokenizer)
        all_auc_drops.append(result["auc_drop"])
        all_curves.append(result["drop_curve"])

    # 计算各遮蔽比例下的平均概率下降值
    if all_curves:
        avg_curve = np.mean(all_curves, axis=0).tolist()
    else:
        avg_curve = [0.0] * len(MASKING_RATIOS)

    mean_auc_drop = float(np.mean(all_auc_drops)) if all_auc_drops else 0.0

    return {
        "method": method_name,
        "dataset": dataset_name,
        "mean_auc_drop": mean_auc_drop,
        "auc_drop_curves": avg_curve,
    }
