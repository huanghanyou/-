"""
注意力权重提取与多头注意力矩阵聚合模块

模块功能：
    从 BertModel 的输出中提取所有层、所有头的注意力权重矩阵，
    并通过聚合策略得到每层对各 token 的注意力分布。

多头注意力聚合策略：
    BERT-base 每层有 12 个注意力头，各头关注输入序列的不同子空间。
    本模块对同一层的 12 个头取平均，得到该层的聚合注意力矩阵。
    聚合后取 [CLS] token（第 0 行）对各 token 的注意力分数，
    反映 [CLS] 表示在构建过程中对各 token 的关注程度。

[CLS] 注意力的含义及局限性：
    [CLS] 对各 token 的注意力分数描述了模型在构建句级表示时的信息聚合模式。
    但注意力权重并不等同于特征重要性：注意力高的 token 不一定对分类结果
    有正面贡献，低注意力的 token 也可能通过间接路径影响输出。因此，
    注意力可视化适合作为模型行为的探索性工具，但不宜作为归因方法的唯一依据。

依赖模块：
    - models/bert_classifier.py：BertTextClassifier 模型
    - config.py：超参数配置

作者：Kris
"""

import torch
import numpy as np
from transformers import BertTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_SEQ_LEN, DEVICE, ATTENTION_TOP_K


def extract_attention_weights(model, input_ids, attention_mask, token_type_ids):
    """
    从 BERT 模型中提取所有层的注意力权重

    参数：
        model: BertTextClassifier 实例
        input_ids: token 编码序列，形状 (1, seq_len)
        attention_mask: 注意力掩码，形状 (1, seq_len)
        token_type_ids: 段落类型编码，形状 (1, seq_len)

    返回值：
        attentions: numpy 数组，形状 (num_layers, num_heads, seq_len, seq_len)
                    BERT-base 中 num_layers=12, num_heads=12
    """
    model.eval()
    with torch.no_grad():
        # 获取注意力权重，返回元组，每层一个张量
        attentions = model.get_attentions(
            input_ids.to(DEVICE),
            attention_mask.to(DEVICE),
            token_type_ids.to(DEVICE)
        )

    # 将各层注意力权重堆叠为单个数组
    # 每层形状: (1, num_heads, seq_len, seq_len) -> 去掉 batch 维度
    attention_array = np.array([att[0].cpu().numpy() for att in attentions])
    return attention_array


def aggregate_attention(attention_array):
    """
    对多头注意力进行聚合，得到各层 [CLS] token 对各位置的注意力分布

    聚合方式：对每层的 12 个注意力头取平均，然后提取 [CLS]（第 0 行）的注意力向量。

    参数：
        attention_array: 形状 (num_layers, num_heads, seq_len, seq_len) 的注意力权重

    返回值：
        layer_attention: 字典，键为层号（字符串），值为 [CLS] 对各 token 的注意力分数列表
                         例如 {"0": [0.1, 0.3, ...], "1": [...], ...}
    """
    num_layers = attention_array.shape[0]
    layer_attention = {}

    for layer_idx in range(num_layers):
        # 对该层的所有注意力头取平均
        # 形状: (num_heads, seq_len, seq_len) -> (seq_len, seq_len)
        avg_attention = attention_array[layer_idx].mean(axis=0)

        # 提取 [CLS] token（第 0 行）对各 token 的注意力分数
        cls_attention = avg_attention[0].tolist()
        layer_attention[str(layer_idx)] = cls_attention

    return layer_attention


def get_top_k_tokens(cls_attention_scores, tokens, k=ATTENTION_TOP_K):
    """
    返回注意力分数最高的 k 个 token 及其分数

    参数：
        cls_attention_scores: [CLS] 对各 token 的注意力分数列表
        tokens: 对应位置的 token 字符串列表
        k: 返回的 top-k 数量

    返回值：
        top_k_list: 列表，每个元素为 {"token": str, "score": float}，按分数降序排列
    """
    # 仅考虑有效 token（排除 [PAD] 等）
    scored_tokens = []
    for i, (token, score) in enumerate(zip(tokens, cls_attention_scores)):
        if token not in ["[PAD]", "[CLS]", "[SEP]"]:
            scored_tokens.append({"token": token, "score": float(score), "position": i})

    # 按分数降序排列
    scored_tokens.sort(key=lambda x: x["score"], reverse=True)

    return scored_tokens[:k]


def explain_attention(model, text, tokenizer=None):
    """
    对单条文本进行注意力权重分析

    参数：
        model: BertTextClassifier 实例
        text: 输入文本字符串
        tokenizer: BertTokenizer 实例，若为 None 则自动加载

    返回值：
        dict: 包含以下字段
            - text: 原始文本
            - tokens: token 字符串列表
            - layer_attention: 各层 [CLS] 注意力分数字典
            - top_k_tokens: 最后一层注意力分数最高的 k 个 token
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 分词
    encoding = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_SEQ_LEN, return_tensors="pt",
        return_token_type_ids=True
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]

    # 获取 token 字符串列表
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # 计算有效序列长度（非 padding 部分）
    valid_len = attention_mask[0].sum().item()

    # 提取注意力权重
    attention_array = extract_attention_weights(
        model, input_ids, attention_mask, token_type_ids
    )

    # 聚合注意力
    layer_attention = aggregate_attention(attention_array)

    # 截断至有效长度
    tokens_valid = tokens[:valid_len]
    layer_attention_valid = {
        k: v[:valid_len] for k, v in layer_attention.items()
    }

    # 获取最后一层的 top-k token
    last_layer_key = str(attention_array.shape[0] - 1)
    top_k = get_top_k_tokens(
        layer_attention_valid[last_layer_key], tokens_valid
    )

    return {
        "text": text,
        "tokens": tokens_valid,
        "layer_attention": layer_attention_valid,
        "top_k_tokens": top_k,
    }
