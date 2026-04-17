"""
Integrated Gradients 归因实现

模块功能：
    使用 captum 库的 IntegratedGradients 方法对 BertTextClassifier 进行
    token 级归因分析。通过计算从基线输入到实际输入路径上的梯度积分，
    量化每个 token 对模型预测结果的贡献程度。

积分步数选取：
    积分步数（n_steps）控制路径积分的近似精度。步数越大，近似越接近
    连续积分的理论值，但计算开销也随之增加。实践中 50-300 步通常足够，
    本项目默认使用 config.IG_N_STEPS 指定的步数。

L2 范数聚合：
    Integrated Gradients 的原始输出为每个 token 对应嵌入维度（768 维）上的
    归因向量。为得到每个 token 的标量归因值，对各 token 的归因向量计算
    L2 范数：score_i = ||attr_i||_2。L2 范数保留了各维度归因的总量信息，
    避免正负值相消导致的信息丢失。

完整性公理：
    Integrated Gradients 满足完整性公理（Completeness Axiom），即所有特征
    的归因值之和等于模型在实际输入与基线输入之间的输出差值。这一性质
    保证归因结果能够完整解释模型的预测决策。

依赖模块：
    - models/bert_classifier.py：BertTextClassifier 模型
    - config.py：超参数配置

作者：Kris
"""

import torch
import numpy as np
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from transformers import BertTokenizer

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_SEQ_LEN, DEVICE, IG_N_STEPS


def _forward_func(input_embeds, attention_mask, token_type_ids, model):
    """
    用于 captum 的前向传播包装函数

    参数：
        input_embeds: 嵌入层的输出，形状 (batch_size, seq_len, hidden_size)
        attention_mask: 注意力掩码
        token_type_ids: 段落类型编码
        model: BertTextClassifier 实例

    返回值：
        logits: 模型输出的 logits
    """
    # 将嵌入直接传入 BERT encoder，绕过嵌入层
    outputs = model.bert(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    cls_hidden = outputs.last_hidden_state[:, 0, :]
    pooled = model.dropout(cls_hidden)
    logits = model.classifier(pooled)
    return logits


def explain_sample(model, text, target_label=None, tokenizer=None):
    """
    对单条文本进行 Integrated Gradients 归因分析

    参数：
        model: BertTextClassifier 实例
        text: 输入文本字符串
        target_label: 归因的目标类别索引。若为 None，使用模型预测的类别。
        tokenizer: BertTokenizer 实例，若为 None 则自动加载

    返回值：
        dict: 包含以下字段
            - text: 原始文本
            - tokens: token 字符串列表（仅有效部分）
            - attribution_scores: 各 token 的归因分数列表
            - predicted_label: 模型预测类别
            - true_label: 未知时为 None
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    # 分词
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    token_type_ids = encoding["token_type_ids"].to(DEVICE)

    # 获取 token 列表
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    valid_len = attention_mask[0].sum().item()

    # 获取模型预测类别
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask, token_type_ids)
        predicted_label = torch.argmax(logits, dim=1).item()

    if target_label is None:
        target_label = predicted_label

    # 获取输入的嵌入表示
    # model.bert.embeddings 将 input_ids 映射为嵌入向量
    input_embeds = model.bert.embeddings(input_ids, token_type_ids=token_type_ids)
    input_embeds = input_embeds.detach().requires_grad_(True)

    # 构造基线嵌入：全零嵌入，对应全 [PAD] token 的输入
    # 全零基线是 IG 中常用的中性参考点
    baseline_embeds = torch.zeros_like(input_embeds)

    # 定义前向函数，固定 attention_mask 和 token_type_ids
    def forward_fn(input_embeds):
        return _forward_func(input_embeds, attention_mask, token_type_ids, model)

    # 创建 IntegratedGradients 对象
    ig = IntegratedGradients(forward_fn)

    try:
        # 计算归因
        # n_steps: 积分路径的离散化步数
        # target: 归因的目标类别
        # 返回形状: (1, seq_len, hidden_size)
        attributions = ig.attribute(
            input_embeds,
            baselines=baseline_embeds,
            target=target_label,
            n_steps=IG_N_STEPS,
            return_convergence_delta=False,
        )
    except Exception as e:
        # 如果 IG 出错，使用简单的梯度作为后备
        print(f"IG计算失败: {e}, 使用简单梯度代替")
        input_embeds.requires_grad_(True)
        with torch.enable_grad():
            logits = forward_fn(input_embeds)
            loss = logits[0, target_label]
            loss.backward()
            attributions = input_embeds.grad.abs()

    # 对嵌入维度计算 L2 范数，得到每个 token 的标量归因值
    # 形状: (1, seq_len, hidden_size) -> (1, seq_len)
    if len(attributions.shape) == 3:
        attr_scores = torch.norm(attributions, dim=-1).squeeze(0)
    else:
        # 处理 attributions 可能是 (seq_len, hidden_size) 的情况
        attr_scores = torch.norm(attributions, dim=-1)

    attr_scores = attr_scores.cpu().detach().numpy()

    # 截断至有效长度
    tokens_valid = tokens[:valid_len]
    scores_valid = attr_scores[:valid_len].tolist()

    return {
        "text": text,
        "tokens": tokens_valid,
        "attribution_scores": scores_valid,
        "predicted_label": predicted_label,
        "true_label": None,
    }
