"""
Grad-CAM 在 Vision Transformer (ViT) 上的适配实现

模块功能：
    实现 Grad-CAM 在 ViT 模型上的适配，使用 HuggingFace 的
    ViTForImageClassification (google/vit-base-patch16-224) 作为目标模型。
    本模块作为独立的演示模块运行，与文本分类主流程相互独立。

Hook 注册与释放机制：
    PyTorch 的 hook 机制允许在不修改模型代码的前提下，在前向或反向传播过程中
    拦截中间层的输入/输出。本模块通过 register_forward_hook 捕获目标层的
    特征图输出，通过 register_full_backward_hook 捕获反向传播中的梯度。
    hook 在使用完毕后必须调用 remove() 方法释放，否则会导致内存泄漏。

Grad-CAM 在 ViT patch token 上的适配方式：
    标准 Grad-CAM 为 CNN 设计，利用卷积层的空间特征图。ViT 将图像分为
    固定大小的 patch（16x16），每个 patch 经线性投影后作为 Transformer 的
    输入 token。ViT 最后一层 Transformer 编码器输出的 token 序列（不含 [CLS]）
    可视为类似 CNN 特征图的空间表示，只是以序列形式排列而非二维网格。
    本模块将这些 token 重新排列为二维网格，然后按通道维度计算梯度权重并加权求和。

与 CNN Grad-CAM 的差异：
    - CNN Grad-CAM 直接利用卷积层输出的空间特征图，空间维度与输入图像有
      对应的降采样关系。
    - ViT Grad-CAM 需要将一维 token 序列重塑为二维空间网格，空间分辨率
      由 patch 数量决定（224/16 = 14，即 14x14）。
    - ViT 的自注意力是全局的，每个 patch token 可关注所有其他 patch，
      而 CNN 的感受野是局部逐层扩展的。

依赖模块：
    - config.py：设备配置

作者：Kris
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import ViTForImageClassification, ViTFeatureExtractor

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIT_MODEL_NAME, DEVICE


class ViTGradCAM:
    """
    ViT 模型的 Grad-CAM 实现

    参数：
        model_name: ViT 预训练模型名称
        device: 计算设备

    属性：
        model: ViTForImageClassification 实例
        feature_extractor: ViT 特征提取器
        target_layer: 目标 Transformer 编码器层（最后一层）
        gradients: 反向传播中捕获的梯度
        activations: 前向传播中捕获的特征图
    """

    def __init__(self, model_name=VIT_MODEL_NAME, device=DEVICE):
        self.device = device

        # 加载预训练 ViT 模型和特征提取器
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

        # 目标层：ViT 最后一层 Transformer 编码器的输出层归一化
        # ViT 的编码器由多个 ViTLayer 组成，取最后一层的 layernorm_after
        self.target_layer = self.model.vit.encoder.layer[-1].layernorm_after

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
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # 捕获反向传播中的梯度
            # grad_output 是一个元组，取第一个元素
            self.gradients = grad_output[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """
        释放已注册的 hook，防止内存泄漏

        在使用完毕后必须调用此方法。
        """
        self._forward_handle.remove()
        self._backward_handle.remove()

    def explain_image(self, image_tensor, target_class=None):
        """
        对输入图像生成 Grad-CAM 激活图

        参数：
            image_tensor: 预处理后的图像张量，形状 (1, 3, 224, 224)
            target_class: 目标类别索引。若为 None，使用模型预测概率最高的类别。

        返回值：
            cam: 归一化至 [0, 1] 的激活图，numpy 数组，形状 (14, 14)
                 14x14 对应 ViT 将 224x224 图像分为 14x14 个 patch

        计算步骤：
            1. 前向传播，通过 hook 捕获目标层的激活值
            2. 对目标类别的 logit 进行反向传播，通过 hook 捕获梯度
            3. 计算各通道的梯度权重 alpha_k = (1/Z) * sum(grad_k)
            4. 加权求和得到 CAM: cam = ReLU(sum(alpha_k * A_k))
            5. 归一化至 [0, 1]
        """
        image_tensor = image_tensor.to(self.device)
        self.model.zero_grad()

        # 前向传播
        outputs = self.model(pixel_values=image_tensor)
        logits = outputs.logits

        # 确定目标类别
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # 反向传播，仅对目标类别的 logit 求梯度
        target_score = logits[0, target_class]
        target_score.backward()

        # 获取 hook 捕获的梯度和激活值
        # 形状: (1, num_tokens+1, hidden_size)，+1 是 [CLS] token
        gradients = self.gradients
        activations = self.activations

        # 去除 [CLS] token（第 0 个位置），仅保留 patch token
        # ViT-base-patch16-224 有 196 个 patch token (14*14)
        patch_gradients = gradients[0, 1:, :]    # (196, 768)
        patch_activations = activations[0, 1:, :]  # (196, 768)

        # 计算各通道（hidden_size 维度）的梯度权重
        # alpha_k = (1/Z) * sum_{i}(grad_{i,k})
        # Z 为 patch 数量 (196)，沿 patch 维度求平均
        alpha = patch_gradients.mean(dim=0)  # (768,)

        # 加权求和：cam = sum_k(alpha_k * A_{:,k})
        # 形状: (196, 768) * (768,) -> (196,) 通过矩阵乘法
        cam = (patch_activations * alpha.unsqueeze(0)).sum(dim=-1)  # (196,)

        # ReLU 激活，仅保留正贡献区域
        cam = F.relu(cam)

        # 重塑为 14x14 空间网格
        num_patches_side = int(cam.shape[0] ** 0.5)  # 14
        cam = cam.reshape(num_patches_side, num_patches_side)

        # 归一化至 [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def __del__(self):
        """析构时释放 hook"""
        try:
            self.remove_hooks()
        except Exception:
            pass


def demo_gradcam():
    """
    Grad-CAM 演示函数

    使用随机生成的图像张量运行 Grad-CAM，展示激活图的计算流程。
    此函数仅用于验证模块功能，不依赖真实图像数据。

    返回值：
        dict: 包含以下字段
            - cam: 14x14 激活图数组
            - predicted_class: 模型预测类别
            - cam_shape: 激活图形状
    """
    # 创建 Grad-CAM 对象
    gradcam = ViTGradCAM()

    # 生成随机图像张量 (1, 3, 224, 224)
    # 模拟 ViT 输入格式：3 通道 224x224 图像
    dummy_image = torch.randn(1, 3, 224, 224)

    # 获取预测类别
    with torch.no_grad():
        dummy_output = gradcam.model(pixel_values=dummy_image.to(DEVICE))
        predicted_class = dummy_output.logits.argmax(dim=1).item()

    # 计算 Grad-CAM 激活图
    cam = gradcam.explain_image(dummy_image, target_class=predicted_class)

    # 释放 hook
    gradcam.remove_hooks()

    print(f"Grad-CAM 演示完成 - 预测类别: {predicted_class}, 激活图形状: {cam.shape}")

    return {
        "cam": cam.tolist(),
        "predicted_class": predicted_class,
        "cam_shape": list(cam.shape),
    }


if __name__ == "__main__":
    demo_gradcam()
