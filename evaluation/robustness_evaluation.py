"""
改进的敏感度和鲁棒性评估模块

模块功能：
    评估遮蔽策略导致的额外干扰，并提供多维度的敏感度分析。

    核心问题：
    - 遮蔽操作本身可能改变文本的语义或语法
    - 不同遮蔽策略对评估结果的影响不同
    - 需要量化遮蔽导致的"虚假信号"

    新增指标：
    1. 噪声影响系数：随机遮蔽相同数量的 token，观察预测变化
    2. 策略稳定性：不同遮蔽策略下结果的一致性
    3. 对抗性评估：使用最坏情况下的遮蔽策略评估
    4. 梯度拉普拉斯平滑性：评估梯度变化的光滑程度

依赖模块：
    - models/bert_classifier.py：BertTextClassifier 模型
    - evaluation/faithfulness_advanced.py：改进的忠实度评估
    - config.py：超参数配置

作者：Kris
"""

import torch
import numpy as np
from transformers import BertTokenizer
from typing import List, Dict, Tuple, Optional
from scipy.stats import spearmanr, kendalltau

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_SEQ_LEN, DEVICE, MASKING_RATIOS


def compute_masking_noise_impact(
    model,
    text: str,
    tokenizer: Optional[BertTokenizer] = None,
    num_trials: int = 10,
) -> Dict:
    """
    评估遮蔽操作本身的噪声影响

    方法：随机遮蔽相同数量的 token，计算预测变化，与有意义的遮蔽结果比较。

    参数：
        model: BertTextClassifier 实例
        text: 原始文本
        tokenizer: BertTokenizer 实例
        num_trials: 随机遮蔽的试验次数

    返回值：
        包含噪声影响分析的字典
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 分词
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]

    # 获取原始预测
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE)
        )
        original_probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(original_probs, dim=1).item()
        original_prob = original_probs[0, predicted_class].item()

    # 计算有效 token 范围
    valid_len = attention_mask[0].sum().item()
    maskable_positions = list(range(1, valid_len - 1))  # 排除 [CLS] 和 [SEP]

    if len(maskable_positions) == 0:
        return {
            "noise_impact_scores": [],
            "mean_noise_impact": 0.0,
            "noise_std": 0.0,
        }

    noise_impacts = []
    mask_token_id = tokenizer.mask_token_id

    for ratio in MASKING_RATIOS:
        num_mask = max(1, int(len(maskable_positions) * ratio))

        ratio_noise_impacts = []
        for _ in range(num_trials):
            # 随机选择位置
            random_positions = np.random.choice(
                maskable_positions, size=num_mask, replace=False
            )

            # 遮蔽
            masked_input_ids = input_ids.clone()
            for pos in random_positions:
                masked_input_ids[0, pos] = mask_token_id

            # 计算预测变化
            with torch.no_grad():
                masked_logits, _ = model(
                    masked_input_ids.to(DEVICE),
                    attention_mask.to(DEVICE),
                    token_type_ids.to(DEVICE),
                )
                masked_probs = torch.softmax(masked_logits, dim=-1)
                masked_prob = masked_probs[0, predicted_class].item()

            noise_impact = original_prob - masked_prob
            ratio_noise_impacts.append(noise_impact)

        mean_noise_impact = float(np.mean(ratio_noise_impacts))
        noise_impacts.append(mean_noise_impact)

    return {
        "noise_impact_scores": noise_impacts,
        "mean_noise_impact": float(np.mean(noise_impacts)),
        "noise_std": float(np.std(noise_impacts)),
    }


def evaluate_strategy_consistency(
    drop_curves_by_strategy: Dict[str, List[float]],
) -> Dict:
    """
    评估不同遮蔽策略的一致性

    使用 Spearman 和 Kendall 相关系数度量不同策略下结果的一致性。

    参数：
        drop_curves_by_strategy: {策略名 -> drop_curve 列表}

    返回值：
        包含一致性分析的字典
    """
    strategy_names = list(drop_curves_by_strategy.keys())

    if len(strategy_names) < 2:
        return {"consistency": "需要至少 2 种策略"}

    consistency_results = {}

    for i, strat1 in enumerate(strategy_names):
        for strat2 in strategy_names[i + 1 :]:
            curve1 = drop_curves_by_strategy[strat1]
            curve2 = drop_curves_by_strategy[strat2]

            # Spearman 相关系数
            spearman_corr, spearman_pval = spearmanr(curve1, curve2)

            # Kendall Tau
            kendall_corr, kendall_pval = kendalltau(curve1, curve2)

            pair_key = f"{strat1}_vs_{strat2}"
            consistency_results[pair_key] = {
                "spearman": float(spearman_corr),
                "spearman_pval": float(spearman_pval),
                "kendall": float(kendall_corr),
                "kendall_pval": float(kendall_pval),
            }

    return consistency_results


def compute_adversarial_faithfulness(
    model,
    text: str,
    attribution_scores: List[float],
    tokenizer: Optional[BertTokenizer] = None,
) -> Dict:
    """
    对抗性忠实度评估

    寻找最容易破坏的 token（而非最重要的 token），评估方法是否过度关注它们。

    参数：
        model: BertTextClassifier 实例
        text: 原始文本
        attribution_scores: token 级归因分数
        tokenizer: BertTokenizer 实例

    返回值：
        包含对抗性评估结果的字典
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 分词
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]

    # 获取原始预测
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE)
        )
        original_probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(original_probs, dim=1).item()
        original_prob = original_probs[0, predicted_class].item()

    # 计算有效 token 数量
    valid_len = attention_mask[0].sum().item()
    num_attributable = min(len(attribution_scores), valid_len - 2)

    if num_attributable <= 0:
        return {
            "adversarial_auc_drop": 0.0,
            "ranking_disagreement": 0.0,
        }

    # 计算每个 token 单独遮蔽时的影响
    individual_impacts = []
    mask_token_id = tokenizer.mask_token_id

    for idx in range(num_attributable):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, idx + 1] = mask_token_id

        with torch.no_grad():
            masked_logits, _ = model(
                masked_input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                token_type_ids.to(DEVICE),
            )
            masked_probs = torch.softmax(masked_logits, dim=-1)
            masked_prob = masked_probs[0, predicted_class].item()

        impact = original_prob - masked_prob
        individual_impacts.append(impact)

    # 按实际影响排序（对抗排序）
    adversarial_ranking = np.argsort(individual_impacts)[::-1]

    # 按归因分数排序
    attribution_ranking = np.argsort(attribution_scores[:num_attributable])[::-1]

    # 计算排序的不一致性（Kendall tau）
    ranking_agreement, _ = kendalltau(adversarial_ranking, attribution_ranking)
    ranking_disagreement = 1.0 - ranking_agreement  # 转换为不一致性

    # 使用对抗排序计算 AUC-Drop
    adversarial_drop_curve = []
    for ratio in MASKING_RATIOS:
        num_mask = max(1, int(num_attributable * ratio))
        positions_to_mask = [int(idx) + 1 for idx in adversarial_ranking[:num_mask]]

        masked_input_ids = input_ids.clone()
        for pos in positions_to_mask:
            if pos < MAX_SEQ_LEN:
                masked_input_ids[0, pos] = mask_token_id

        with torch.no_grad():
            masked_logits, _ = model(
                masked_input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                token_type_ids.to(DEVICE),
            )
            masked_probs = torch.softmax(masked_logits, dim=-1)
            masked_prob = masked_probs[0, predicted_class].item()

        drop = original_prob - masked_prob
        adversarial_drop_curve.append(float(drop))

    adversarial_auc_drop = float(np.trapz(adversarial_drop_curve, MASKING_RATIOS))

    return {
        "adversarial_auc_drop": adversarial_auc_drop,
        "ranking_disagreement": float(ranking_disagreement),
        "adversarial_drop_curve": adversarial_drop_curve,
    }


def compute_faithfulness_robustness_score(
    original_auc_drop: float,
    adversarial_auc_drop: float,
    noise_impact: float,
) -> Dict:
    """
    综合计算忠实度的鲁棒性分数

    综合考虑：
    1. 原始忠实度（越高越好）
    2. 对抗性忠实度（与原始相近越好）
    3. 噪声鲁棒性（不受随机遮蔽影响越好）

    参数：
        original_auc_drop: 原始 AUC-Drop
        adversarial_auc_drop: 对抗性 AUC-Drop
        noise_impact: 平均噪声影响

    返回值：
        包含鲁棒性分数的字典
    """
    # 鲁棒性分数 = 原始忠实度 * 对抗稳定性 * 噪声抵抗性

    # 对抗稳定性：对抗排序与原始排序越相似越好
    # 如果 adversarial_auc_drop 远小于 original_auc_drop，说明方法找到的重要 token 不是最易破坏的
    adversarial_stability = (
        min(1.0, adversarial_auc_drop / original_auc_drop)
        if original_auc_drop > 0
        else 0.0
    )

    # 噪声抵抗性：噪声影响越小越好
    noise_resistance = (
        1.0 - min(1.0, noise_impact / original_auc_drop)
        if original_auc_drop > 0
        else 1.0
    )

    # 综合鲁棒性分数
    robustness_score = (
        original_auc_drop
        * adversarial_stability
        * (1 - noise_impact / max(original_auc_drop, 1e-6))
    )
    robustness_score = max(0, min(1, robustness_score))  # 限制在 [0, 1]

    return {
        "original_auc_drop": original_auc_drop,
        "adversarial_auc_drop": adversarial_auc_drop,
        "adversarial_stability": float(adversarial_stability),
        "noise_impact": noise_impact,
        "noise_resistance": float(noise_resistance),
        "robustness_score": float(robustness_score),
    }
