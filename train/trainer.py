"""
通用训练循环模块

模块功能：
    实现适用于 SST-2 和 CWRU 两个数据集的通用训练循环。
    使用 AdamW 优化器配合线性 warmup 学习率调度策略。
    每个 epoch 结束后在验证集上进行评估，并保存验证集 F1 最高的模型权重。

Warmup 调度说明：
    线性 warmup 策略在训练初始阶段将学习率从 0 线性增加至目标学习率，
    随后线性衰减至 0。warmup 阶段有助于在训练初期避免梯度过大导致的
    参数剧烈更新，从而提升微调预训练模型的稳定性。

最佳模型保存策略：
    以验证集上的 macro F1 分数作为模型选择标准，每当当前 epoch 的 F1
    超过历史最佳值时，保存模型权重至 models/saved/ 目录。

依赖模块：
    - config.py：超参数配置
    - train/evaluator.py：验证集评估

作者：Kris
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LEARNING_RATE, NUM_EPOCHS, DEVICE, MODEL_SAVE_DIR, WARMUP_RATIO
)
from train.evaluator import evaluate_model


def train_model(model, train_loader, val_loader, dataset_name="sst2"):
    """
    训练文本分类模型

    参数：
        model: BertTextClassifier 实例
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        dataset_name: 数据集名称，用于保存模型文件命名，"sst2" 或 "cwru"

    返回值：
        model: 训练完成并加载了最佳权重的模型
        best_metrics: 最佳验证集上的评估指标字典

    训练流程：
        1. 初始化 AdamW 优化器和线性 warmup 学习率调度器
        2. 遍历每个 epoch，执行前向传播、损失计算、反向传播、参数更新
        3. 每个 epoch 结束后在验证集上评估
        4. 保存 F1 最高的模型权重
        5. 训练结束后加载最佳权重
    """
    model = model.to(DEVICE)

    # 使用 AdamW 优化器，对 BERT 参数进行微调
    # weight_decay 用于 L2 正则化，减少过拟合风险
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 计算总训练步数和 warmup 步数
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # 线性 warmup 调度器：前 warmup_steps 步线性增加学习率，之后线性衰减
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 最佳模型跟踪
    best_f1 = 0.0
    best_metrics = None

    # 确保模型保存目录存在
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{dataset_name}.pt")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # 使用 tqdm 显示训练进度
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=True
        )

        for batch in progress_bar:
            # 将数据移至计算设备
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # 前向传播
            logits, _ = model(input_ids, attention_mask, token_type_ids)

            # 计算交叉熵损失
            loss = criterion(logits, labels)

            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条显示的损失值
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - 平均训练损失: {avg_loss:.4f}")

        # 在验证集上评估
        val_metrics = evaluate_model(model, val_loader)
        print(
            f"验证集 - Accuracy: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, "
            f"Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}"
        )

        # 保存最佳模型（以 F1 为选择标准）
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), save_path)
            print(f"最佳模型已保存至 {save_path}，F1: {best_f1:.4f}")

    # 训练结束后加载最佳模型权重
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f"训练完成，已加载最佳模型权重（F1: {best_f1:.4f}）")

    return model, best_metrics


def load_trained_model(model, dataset_name="sst2"):
    """
    加载已训练的模型权重

    参数：
        model: BertTextClassifier 实例（未加载权重）
        dataset_name: 数据集名称，"sst2" 或 "cwru"

    返回值：
        model: 加载了训练权重的模型
    """
    save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{dataset_name}.pt")
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"未找到训练好的模型权重文件: {save_path}。请先运行训练。"
        )
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"已加载 {dataset_name} 模型权重: {save_path}")
    return model
