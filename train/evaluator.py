"""
模型评估模块

模块功能：
    在测试集或验证集上计算分类性能指标，包括 accuracy、macro precision、
    macro recall 和 macro F1。同时返回预测标签与真实标签列表，供结果保存模块使用。

指标计算方式说明：
    - Accuracy: 预测正确的样本数 / 总样本数
    - Macro Precision: 各类别 Precision 的算术平均
    - Macro Recall: 各类别 Recall 的算术平均
    - Macro F1: 各类别 F1 的算术平均
    Macro 平均对各类别赋予相同权重，适用于类别均衡的数据集。

依赖模块：
    - config.py：设备配置

作者：Kris
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE


def evaluate_model(model, data_loader):
    """
    在给定数据集上评估模型性能

    参数：
        model: 训练好的 BertTextClassifier 实例
        data_loader: 测试集或验证集的 DataLoader

    返回值：
        dict: 包含以下键值对的评估结果字典
            - accuracy (float): 准确率
            - precision (float): macro 精确率
            - recall (float): macro 召回率
            - f1 (float): macro F1
            - per_class_f1 (dict): 各类别的 F1 分数
            - predictions (list): 预测标签列表
            - true_labels (list): 真实标签列表
    """
    model = model.to(DEVICE)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # 前向传播获取 logits
            logits, _ = model(input_ids, attention_mask, token_type_ids)

            # 取 argmax 得到预测类别
            preds = torch.argmax(logits, dim=1)

            all_predictions.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # 计算各项指标
    accuracy = accuracy_score(all_labels, all_predictions)

    # macro 平均：对每个类别分别计算指标后取算术平均
    # zero_division=0 表示当某类别无预测样本时，该类别指标记为 0
    precision = precision_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )

    # 各类别的 F1 分数
    per_class_f1_array = f1_score(
        all_labels, all_predictions, average=None, zero_division=0
    )
    unique_labels = sorted(set(all_labels))
    per_class_f1 = {
        str(label): float(score)
        for label, score in zip(unique_labels, per_class_f1_array)
    }

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_class_f1": per_class_f1,
        "predictions": all_predictions,
        "true_labels": all_labels,
    }

    return results
