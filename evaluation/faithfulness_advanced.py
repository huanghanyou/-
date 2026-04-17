"""
改进的忠实度评估模块 - 考虑语言特性的多策略评估

模块功能：
    针对教育文本的特性，实现多种遮蔽策略和评估指标，以减少语言干扰对评估结果的影响。

问题分析：
    原始遮蔽策略的局限：
    1. [MASK] 遮蔽：可能破坏句子语法结构，如主谓搭配、时态一致等
    2. 不考虑词性：删除动词和删除介词对句子的影响不同
    3. 不考虑位置：删除开头和删除末尾的影响不同
    4. 教育文本特性：包含术语、列举结构、复杂句式等特殊结构

改进策略：
    1. 多种遮蔽方法：替换、删除、随机化、上下文感知替换
    2. 词性感知：对不同词性分别评估
    3. 上下文保留：使用语言模型预测的替换词而非 [MASK]
    4. 评估指标多元化：不仅关注概率变化，还关注预测类别稳定性
    5. 噪声评估：量化遮蔽导致的额外干扰

依赖模块：
    - models/bert_classifier.py：BertTextClassifier 模型
    - config.py：超参数配置

作者：Kris
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from typing import List, Dict, Tuple, Optional
from enum import Enum

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_SEQ_LEN, DEVICE, MASKING_RATIOS


class MaskingStrategy(Enum):
    """遮蔽策略枚举"""

    MASK_TOKEN = "mask_token"  # 替换为 [MASK]
    DELETION = "deletion"  # 直接删除 token
    RANDOM_TOKEN = "random_token"  # 替换为随机 token
    PADDING = "padding"  # 替换为 [PAD]
    CONTEXTUAL = "contextual"  # 使用 LM 预测的 token 替换


class POSTag(Enum):
    """词性标注（简化版，基于 token 特征判断）"""

    VERB = "verb"
    NOUN = "noun"
    ADJECTIVE = "adjective"
    ARTICLE = "article"
    PREPOSITION = "preposition"
    PUNCTUATION = "punctuation"
    OTHER = "other"


def _get_prediction_prob(
    model, input_ids, attention_mask, token_type_ids, target_class
):
    """
    获取模型对指定类别的预测概率
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE)
        )
        probs = torch.softmax(logits, dim=-1)
    return probs[0, target_class].item()


def _estimate_pos_tag(token: str) -> POSTag:
    """
    简化的词性估计（基于 token 特征）

    在实际应用中，应使用专业的 POS tagger
    """
    token_lower = token.lower()

    # 文章
    if token_lower in ["a", "an", "the"]:
        return POSTag.ARTICLE

    # 介词
    if token_lower in ["in", "on", "at", "to", "from", "by", "with", "for", "of", "as"]:
        return POSTag.PREPOSITION

    # 标点
    if token in [".", ",", "!", "?", ";", ":", '"', "'", "-", "(", ")"]:
        return POSTag.PUNCTUATION

    # 常见动词词尾
    if token_lower.endswith(("ing", "ed")) or token_lower in [
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
    ]:
        return POSTag.VERB

    # 常见形容词词尾
    if token_lower.endswith(("ful", "less", "able", "ible")) or token_lower in [
        "good",
        "bad",
        "large",
    ]:
        return POSTag.ADJECTIVE

    return POSTag.OTHER


class AdaptiveMasker:
    """
    自适应遮蔽器：根据策略对 token 进行遮蔽处理
    """

    def __init__(
        self,
        tokenizer: BertTokenizer,
        strategy: MaskingStrategy = MaskingStrategy.MASK_TOKEN,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # 初始化 MLM 模型用于上下文感知替换
        if strategy == MaskingStrategy.CONTEXTUAL:
            self.mlm_model = BertForMaskedLM.from_pretrained(MODEL_NAME)
            self.mlm_model.to(DEVICE)
            self.mlm_model.eval()
        else:
            self.mlm_model = None

    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        positions: List[int],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据遮蔽策略对指定位置的 token 进行处理

        参数：
            input_ids: token 编码序列，形状 (1, seq_len)
            positions: 要遮蔽的位置索引列表
            attention_mask: 注意力掩码

        返回值：
            处理后的 input_ids
        """
        masked_input_ids = input_ids.clone()

        if self.strategy == MaskingStrategy.MASK_TOKEN:
            # 替换为 [MASK]
            for pos in positions:
                if pos < MAX_SEQ_LEN:
                    masked_input_ids[0, pos] = self.mask_token_id

        elif self.strategy == MaskingStrategy.DELETION:
            # 删除 token（实现为替换为 [PAD] 并保持 attention mask）
            for pos in positions:
                if pos < MAX_SEQ_LEN:
                    masked_input_ids[0, pos] = self.pad_token_id

        elif self.strategy == MaskingStrategy.RANDOM_TOKEN:
            # 替换为随机 token
            for pos in positions:
                if pos < MAX_SEQ_LEN:
                    random_token_id = torch.randint(0, len(self.tokenizer), (1,)).item()
                    masked_input_ids[0, pos] = random_token_id

        elif self.strategy == MaskingStrategy.PADDING:
            # 替换为 [PAD]
            for pos in positions:
                if pos < MAX_SEQ_LEN:
                    masked_input_ids[0, pos] = self.pad_token_id

        elif self.strategy == MaskingStrategy.CONTEXTUAL:
            # 使用 MLM 模型预测上下文中的 token
            with torch.no_grad():
                mlm_masked = input_ids.clone()
                for pos in positions:
                    if pos < MAX_SEQ_LEN:
                        mlm_masked[0, pos] = self.mask_token_id

                # 获取 MLM 预测
                outputs = self.mlm_model(
                    mlm_masked.to(DEVICE), attention_mask=attention_mask.to(DEVICE)
                )
                predictions = outputs.logits

                # 取概率最高的非 [MASK] token
                for pos in positions:
                    if pos < MAX_SEQ_LEN:
                        logits = predictions[0, pos]
                        # 排除 [MASK] 和特殊 token
                        logits[self.mask_token_id] = -float("inf")
                        logits[self.tokenizer.cls_token_id] = -float("inf")
                        logits[self.tokenizer.sep_token_id] = -float("inf")

                        predicted_token_id = torch.argmax(logits).item()
                        masked_input_ids[0, pos] = predicted_token_id

        return masked_input_ids


def compute_comprehensiveness_advanced(
    model,
    text: str,
    attribution_scores: List[float],
    tokenizer: Optional[BertTokenizer] = None,
    strategy: MaskingStrategy = MaskingStrategy.MASK_TOKEN,
    analyze_by_pos: bool = True,
) -> Dict:
    """
    计算改进的 Comprehensiveness 指标

    参数：
        model: BertTextClassifier 实例
        text: 原始文本
        attribution_scores: token 级归因分数
        tokenizer: BertTokenizer 实例
        strategy: 遮蔽策略
        analyze_by_pos: 是否按词性分别分析

    返回值：
        包含多个评估指标的字典
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
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

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
            "drop_curve": [0.0] * len(MASKING_RATIOS),
            "auc_drop": 0.0,
            "class_flip_rate": 0.0,
            "by_pos": {},
        }

    # 按归因分数降序排列 token 索引
    sorted_indices = np.argsort(attribution_scores[:num_attributable])[::-1]
    sorted_positions = [int(idx) + 1 for idx in sorted_indices]

    # 初始化自适应遮蔽器
    masker = AdaptiveMasker(tokenizer, strategy)

    drop_curve = []
    class_flips = []

    for ratio in MASKING_RATIOS:
        num_mask = max(1, int(num_attributable * ratio))
        positions_to_mask = sorted_positions[:num_mask]

        # 应用遮蔽策略
        masked_input_ids = masker.mask_tokens(
            input_ids, positions_to_mask, attention_mask
        )

        # 计算遮蔽后的预测概率
        with torch.no_grad():
            masked_logits, _ = model(
                masked_input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                token_type_ids.to(DEVICE),
            )
            masked_probs = torch.softmax(masked_logits, dim=-1)
            masked_prob = masked_probs[0, predicted_class].item()
            masked_class = torch.argmax(masked_probs, dim=1).item()

        # 概率下降
        drop = original_prob - masked_prob
        drop_curve.append(float(drop))

        # 预测类别是否改变（类别翻转）
        class_flip = 1.0 if masked_class != predicted_class else 0.0
        class_flips.append(class_flip)

    # AUC-Drop
    auc_drop = float(np.trapz(drop_curve, MASKING_RATIOS))

    # 类别翻转率（在某个遮蔽比例下预测类别改变的比例）
    class_flip_rate = float(np.mean(class_flips))

    result = {
        "drop_curve": drop_curve,
        "auc_drop": auc_drop,
        "class_flip_rate": class_flip_rate,
        "strategy": strategy.value,
        "by_pos": {},
    }

    # 按词性分别分析
    if analyze_by_pos:
        pos_groups = {}
        for idx, pos_idx in enumerate(sorted_indices):
            token = tokens[pos_idx + 1] if pos_idx + 1 < len(tokens) else ""
            pos_tag = _estimate_pos_tag(token)

            if pos_tag not in pos_groups:
                pos_groups[pos_tag] = []
            pos_groups[pos_tag].append((pos_idx, attribution_scores[pos_idx]))

        # 为各词性组计算忠实度
        for pos_tag, indices_scores in pos_groups.items():
            if not indices_scores:
                continue

            pos_drop_curve = []
            for ratio in MASKING_RATIOS:
                num_mask = max(1, int(len(indices_scores) * ratio))
                pos_positions = [int(idx) + 1 for idx, _ in indices_scores[:num_mask]]

                masked_input_ids = masker.mask_tokens(
                    input_ids, pos_positions, attention_mask
                )

                with torch.no_grad():
                    masked_logits, _ = model(
                        masked_input_ids.to(DEVICE),
                        attention_mask.to(DEVICE),
                        token_type_ids.to(DEVICE),
                    )
                    masked_probs = torch.softmax(masked_logits, dim=-1)
                    masked_prob = masked_probs[0, predicted_class].item()

                drop = original_prob - masked_prob
                pos_drop_curve.append(float(drop))

            pos_auc_drop = float(np.trapz(pos_drop_curve, MASKING_RATIOS))
            result["by_pos"][pos_tag.value] = {
                "auc_drop": pos_auc_drop,
                "count": len(indices_scores),
            }

    return result


def evaluate_faithfulness_advanced(
    model,
    texts: List[str],
    attribution_results: List[Dict],
    method_name: str,
    dataset_name: str,
    tokenizer: Optional[BertTokenizer] = None,
    strategies: Optional[List[MaskingStrategy]] = None,
) -> Dict:
    """
    改进的忠实度评估，支持多种遮蔽策略

    参数：
        model: BertTextClassifier 实例
        texts: 文本列表
        attribution_results: 归因结果列表
        method_name: 方法名称
        dataset_name: 数据集名称
        tokenizer: BertTokenizer 实例
        strategies: 遮蔽策略列表，默认使用所有策略

    返回值：
        包含多个遮蔽策略评估结果的字典
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    if strategies is None:
        strategies = [
            MaskingStrategy.MASK_TOKEN,
            MaskingStrategy.DELETION,
            MaskingStrategy.RANDOM_TOKEN,
            MaskingStrategy.CONTEXTUAL,
        ]

    results_by_strategy = {}

    for strategy in strategies:
        all_auc_drops = []
        all_curves = []
        all_class_flips = []
        all_by_pos = []

        for text, attr_result in zip(texts, attribution_results):
            scores = attr_result.get("attribution_scores", [])
            if not scores:
                continue

            result = compute_comprehensiveness_advanced(
                model,
                text,
                scores,
                tokenizer,
                strategy=strategy,
                analyze_by_pos=True,
            )

            all_auc_drops.append(result["auc_drop"])
            all_curves.append(result["drop_curve"])
            all_class_flips.append(result["class_flip_rate"])
            if result["by_pos"]:
                all_by_pos.append(result["by_pos"])

        # 汇总结果
        if all_curves:
            avg_curve = np.mean(all_curves, axis=0).tolist()
        else:
            avg_curve = [0.0] * len(MASKING_RATIOS)

        mean_auc_drop = float(np.mean(all_auc_drops)) if all_auc_drops else 0.0
        mean_class_flip = float(np.mean(all_class_flips)) if all_class_flips else 0.0

        # 按词性汇总
        pos_summary = {}
        if all_by_pos:
            for pos_tag in POSTag:
                auc_drops = []
                counts = []
                for by_pos in all_by_pos:
                    if pos_tag.value in by_pos:
                        auc_drops.append(by_pos[pos_tag.value]["auc_drop"])
                        counts.append(by_pos[pos_tag.value]["count"])

                if auc_drops:
                    pos_summary[pos_tag.value] = {
                        "mean_auc_drop": float(np.mean(auc_drops)),
                        "total_count": sum(counts),
                    }

        results_by_strategy[strategy.value] = {
            "mean_auc_drop": mean_auc_drop,
            "auc_drop_curves": avg_curve,
            "class_flip_rate": mean_class_flip,
            "by_pos": pos_summary,
        }

    return {
        "method": method_name,
        "dataset": dataset_name,
        "strategies": results_by_strategy,
    }


def compare_strategies_robustness(
    faithfulness_results: Dict,
) -> Dict:
    """
    比较不同遮蔽策略的鲁棒性

    评估指标：
    - 策略一致性：不同策略下 AUC-Drop 的相关性
    - 策略敏感性：不同策略下 AUC-Drop 的变化幅度

    参数：
        faithfulness_results: 包含多策略评估结果的字典

    返回值：
        策略鲁棒性分析结果
    """
    strategies = faithfulness_results.get("strategies", {})

    if len(strategies) < 2:
        return {"robustness_analysis": "需要至少 2 种策略进行比较"}

    auc_drops = {}
    for strategy_name, result in strategies.items():
        auc_drops[strategy_name] = result["mean_auc_drop"]

    # 计算策略间的差异
    auc_values = list(auc_drops.values())
    std_dev = float(np.std(auc_values))
    mean_auc = float(np.mean(auc_values))
    variation_coef = std_dev / mean_auc if mean_auc > 0 else 0

    return {
        "auc_drops_by_strategy": auc_drops,
        "std_dev": std_dev,
        "mean_auc": mean_auc,
        "variation_coefficient": variation_coef,
        "robustness_interpretation": (
            "低" if variation_coef > 0.3 else "中" if variation_coef > 0.15 else "高"
        ),
    }
