"""
Grad-CAM 在 BERT 文本分类上的实现

模块功能：
    实现 Grad-CAM 在 BERT 模型上的适配，用于生成文本分类的属性分数。
    通过计算各层 token 表示对分类输出的梯度权重，得到每个 token 对预测结果的贡献。

Grad-CAM 在 BERT 上的适配方式：
    标准 Grad-CAM 为 CNN 设计，利用卷积层的空间特征图。BERT 是基于 Transformer 的序列模型，
    其各层输出为 token 表示向量（hidden states），形状为 (batch_size, seq_len, hidden_size)。

    BERT Grad-CAM 的思路是：
    1. 选择目标层（通常是最后一层的 [CLS] token 或所有 token 的平均）
    2. 计算该层输出对分类 logit 的梯度
    3. 对各 token 的梯度进行加权，得到每个 token 的重要性分数
    4. 归一化后作为归因结果

与 CNN Grad-CAM 的差异：
    - CNN Grad-CAM 利用卷积层输出的空间特征图（高 × 宽 × 通道）
    - BERT Grad-CAM 直接对 token 表示向量计算梯度权重
    - BERT 的自注意力是全局的，每个 token 可关注所有其他 token，而 CNN 的感受野是局部逐层扩展的
    - 结果是对序列中每个 token 的标量权重，而不是空间热力图

依赖模块：
    - models/bert_classifier.py：BertTextClassifier 模型
    - config.py：设备配置

作者：Kris
"""

import torch
import torch.nn.functional as F
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE


class BertGradCAM:
    """
    BERT 模型的 Grad-CAM 实现

    参数：
        model: BertTextClassifier 实例
        device: 计算设备

    属性：
        model: BERT 分类模型
        target_layer_name: 目标层名称（默认为 "bert.encoder.layer.11"，即最后一层）
        gradients: 反向传播中捕获的梯度
        activations: 前向传播中捕获的特征图
    """

    def __init__(self, model, target_layer_name="bert.encoder.layer.11", device=DEVICE):
        self.model = model
        # 保持训练模式以计算梯度，但不更新 batch norm 统计
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True
        self.device = device
        self.target_layer_name = target_layer_name

        # 存储 hook 捕获的梯度和激活值
        self.gradients = None
        self.activations = None

        # 注册 hook
        self._register_hooks()

    def _register_hooks(self):
        """
        注册前向和反向传播 hook

        前向 hook 捕获目标层的输出（激活值），
        反向 hook 捕获目标层输出对应的梯度。
        """

        def forward_hook(module, input, output):
            # 捕获目标层的输出
            # BERT encoder 层的输出是 tuple，第一个元素是 hidden states
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # 捕获反向传播中的梯度
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        # 获取目标层模块
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_module = module
                break

        if target_module is None:
            raise ValueError(
                f"Target layer '{self.target_layer_name}' not found in model"
            )

        self._forward_handle = target_module.register_forward_hook(forward_hook)
        self._backward_handle = target_module.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """
        释放已注册的 hook，防止内存泄漏

        在使用完毕后必须调用此方法。
        """
        self._forward_handle.remove()
        self._backward_handle.remove()

    def explain_sample(
        self, input_ids, attention_mask, token_type_ids, target_class=None
    ):
        """
        对输入文本生成 Grad-CAM 属性分数

        参数：
            input_ids: token 编码序列，形状 (1, seq_len)
            attention_mask: 注意力掩码，形状 (1, seq_len)
            token_type_ids: 段落类型编码，形状 (1, seq_len)
            target_class: 目标类别索引。若为 None，使用模型预测概率最高的类别。

        返回值：
            dict: 包含以下字段
                - "token_gradcam_scores": list，每个 token 的 Grad-CAM 分数
                - "target_class": int，目标类别索引
                - "predicted_class": int，模型预测的类别
                - "input_ids": list，输入 token ID
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)

        self.model.zero_grad()

        # 前向传播（需要追踪梯度）
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # 确定目标类别
        predicted_class = logits.argmax(dim=1).detach().item()
        if target_class is None:
            target_class = predicted_class

        # 反向传播，仅对目标类别的 logit 求梯度
        target_score = logits[0, target_class]
        target_score.backward()

        # 获取 hook 捕获的梯度和激活值
        # 形状: (1, seq_len, hidden_size)
        gradients = self.gradients  # (1, seq_len, hidden_size)
        activations = self.activations  # (1, seq_len, hidden_size)

        # 计算 Grad-CAM
        # 方法 1：沿 hidden_size 维度对梯度求平均，得到每个 token 的权重
        # alpha_i = (1/C) * sum_c(grad_{i,c})，其中 C 是 hidden_size
        token_weights = gradients[0].mean(dim=1)  # (seq_len,)

        # ReLU 激活，仅保留正贡献
        token_weights = F.relu(token_weights)

        # 归一化
        token_weights = token_weights.cpu().detach().numpy()
        if token_weights.max() > token_weights.min():
            token_weights = (token_weights - token_weights.min()) / (
                token_weights.max() - token_weights.min()
            )

        return {
            "token_gradcam_scores": token_weights.tolist(),
            "target_class": int(target_class),
            "predicted_class": int(predicted_class),
            "input_ids": input_ids[0].cpu().tolist(),
        }


def explain_sample(model, text, tokenizer):
    """
    对单条文本进行 Grad-CAM 分析

    参数：
        model: BertTextClassifier 实例
        text: 输入文本字符串
        tokenizer: BertTokenizer 实例

    返回值：
        dict: 包含以下字段
            - "text": 原始输入文本
            - "tokens": list，token 字符串列表
            - "token_gradcam_scores": list，每个 token 的 Grad-CAM 分数
            - "predicted_class": int，模型预测的类别
            - "target_class": int，目标类别（与预测类别相同）
    """
    # Tokenize 文本
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded["token_type_ids"]

    # 创建 Grad-CAM 对象
    gradcam = BertGradCAM(model, device=DEVICE)

    # 计算 Grad-CAM
    result = gradcam.explain_sample(input_ids, attention_mask, token_type_ids)

    # 释放 hook
    gradcam.remove_hooks()

    # 获取 token 字符串
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    return {
        "text": text,
        "tokens": tokens,
        "token_gradcam_scores": result["token_gradcam_scores"],
        "predicted_class": result["predicted_class"],
        "target_class": result["target_class"],
    }
