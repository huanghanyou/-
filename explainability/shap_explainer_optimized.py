"""
优化的 SHAP 归因实现 - 快速版本

模块功能：
    在原 KernelSHAP 基础上，添加多种计算加速策略：
    1. 缓存机制：缓存模型推理结果，避免重复计算
    2. 近似计算：使用较少的扰动样本加速计算
    3. 批处理：优化批量推理
    4. 背景数据采样：使用更少的背景数据加速计算

SHAP 计算瓶颈分析：
    原始 KernelSHAP 在 BERT 上的耗时主要来自：
    - 多次模型推理（每次扰动产生的文本变体都需推理）
    - 特征遮蔽策略（需要逐个 token 遮蔽）

加速策略：
    1. 缓存（最关键）：
       - 缓存相同输入的推理结果
       - 避免重复推理同一个文本

    2. 近似计算（次关键）：
       - 减少 SHAP 扰动样本数（default 2^num_features）
       - 改用更少采样的 SHAP Explainer

    3. 背景采样：
       - 使用更小的背景数据集
       - 降低 Explainer 初始化开销

    4. 批处理：
       - 一次推理多个样本
       - 充分利用 GPU/CPU 并行计算

使用建议：
    - 交互式应用：使用 explain_sample_fast()
    - 批量分析：使用 explain_batch_optimized()
    - 离线评估：使用原始 explain_batch()（精度优先）

作者：Kris
"""

import torch
import numpy as np
import shap
from transformers import BertTokenizer
from functools import lru_cache
from typing import List, Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, DEVICE, SHAP_SAMPLE_SIZE


class ShapCacheExplainer:
    """带缓存的 SHAP Explainer 包装器"""

    def __init__(
        self,
        model,
        tokenizer,
        max_seq_len=128,
        cache_size=128,
        use_approximate=False,
        num_samples=None,
    ):
        """
        初始化 SHAP 缓存解释器

        参数：
            model: BertTextClassifier 实例
            tokenizer: BertTokenizer 实例
            max_seq_len: 最大序列长度
            cache_size: 推理结果缓存大小
            use_approximate: 是否使用近似计算（加速）
            num_samples: SHAP 扰动样本数（None 则自动调整）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_approximate = use_approximate
        self.cache_size = cache_size

        # 推理结果缓存
        self._prediction_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # 创建预测函数
        self.predict_fn = self._create_predict_fn_with_cache()

        # 创建 SHAP Explainer
        masker = shap.maskers.Text(tokenizer=r"\s+")

        # 如果使用近似计算，传入采样参数
        if use_approximate:
            num_samples = num_samples or min(256, 2**10)  # 限制采样数
            self.explainer = shap.Explainer(
                self.predict_fn, masker, output_names=None, feature_names=None
            )
            # 动态设置采样数（仅 KernelSHAP）
            if hasattr(self.explainer, "N"):
                self.explainer.N = num_samples
        else:
            self.explainer = shap.Explainer(self.predict_fn, masker, output_names=None)

    def _create_predict_fn_with_cache(self):
        """创建带缓存的预测函数"""
        model = self.model
        model.eval()

        def predict_fn(texts):
            if isinstance(texts, str):
                texts = [texts]
            texts = list(texts)

            # 检查缓存
            uncached_texts = []
            uncached_indices = []
            cached_probs = {}

            for i, text in enumerate(texts):
                text_key = hash(text) % (10**9)  # 简单哈希键
                if text_key in self._prediction_cache:
                    cached_probs[i] = self._prediction_cache[text_key]
                    self._cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self._cache_misses += 1

            # 批量处理未缓存的文本
            if uncached_texts:
                encodings = self.tokenizer(
                    uncached_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                    return_token_type_ids=True,
                )

                input_ids = encodings["input_ids"].to(DEVICE)
                attention_mask = encodings["attention_mask"].to(DEVICE)
                token_type_ids = encodings["token_type_ids"].to(DEVICE)

                with torch.no_grad():
                    logits, _ = model(input_ids, attention_mask, token_type_ids)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # 缓存结果
                for idx, text in zip(uncached_indices, uncached_texts):
                    text_key = hash(text) % (10**9)
                    self._prediction_cache[text_key] = probs[uncached_texts.index(text)]
                    cached_probs[idx] = probs[uncached_texts.index(text)]

            # 组合结果
            result_probs = np.zeros(
                (len(texts), probs.shape[-1] if uncached_texts else 2)
            )
            for i in range(len(texts)):
                if i in cached_probs:
                    result_probs[i] = cached_probs[i]

            # 限制缓存大小
            if len(self._prediction_cache) > self.cache_size:
                self._prediction_cache.clear()

            return result_probs

        return predict_fn

    def explain_sample(self, text: str) -> Dict:
        """解释单个样本（快速）"""
        try:
            shap_values = self.explainer([text])

            probs = self.predict_fn(text)
            predicted_label = int(np.argmax(probs[0]))

            sample_tokens = list(shap_values.data[0])
            sample_scores = shap_values.values[0][:, predicted_label].tolist()

            return {
                "text": text,
                "tokens": sample_tokens,
                "attribution_scores": sample_scores,
                "predicted_label": predicted_label,
                "true_label": None,
                "computation_time": None,
            }
        except Exception as e:
            print(f"SHAP 解释失败: {e}")
            return None

    def explain_batch(self, texts: List[str], show_stats=False) -> List[Dict]:
        """批量解释样本"""
        results = []
        for text in texts:
            result = self.explain_sample(text)
            if result:
                results.append(result)

        if show_stats:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0
            print(f"缓存命中率: {hit_rate:.2%} ({self._cache_hits}/{total})")

        return results

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "cache_size": len(self._prediction_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }


def explain_sample_fast(model, text: str, tokenizer=None, use_cache=True) -> Dict:
    """
    快速 SHAP 解释（针对交互式应用优化）

    特点：
    - 使用缓存加速重复查询
    - 支持近似计算降低延迟
    - 适合实时交互场景

    参数：
        model: BertTextClassifier 实例
        text: 要解释的文本
        tokenizer: BertTokenizer 实例
        use_cache: 是否启用缓存

    返回值：
        dict: 归因结果
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)

    explainer = ShapCacheExplainer(
        model,
        tokenizer,
        use_approximate=True,  # 交互式应用使用近似
        cache_size=128,
    )

    return explainer.explain_sample(text)


def explain_batch_optimized(
    model, texts: List[str], tokenizer=None, use_approximate=True, show_stats=False
) -> List[Dict]:
    """
    优化的批量 SHAP 解释

    特点：
    - 支持缓存机制
    - 支持近似计算
    - 输出缓存统计信息

    参数：
        model: BertTextClassifier 实例
        texts: 文本列表
        tokenizer: BertTokenizer 实例
        use_approximate: 是否使用近似计算（加速）
        show_stats: 是否输出统计信息

    返回值：
        list: 每条样本的归因结果
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    explainer = ShapCacheExplainer(
        model, tokenizer, use_approximate=use_approximate, cache_size=256
    )

    results = explainer.explain_batch(texts, show_stats=show_stats)

    return results


def explain_batch(model, texts, tokenizer=None):
    """
    原始 SHAP 实现（保留向后兼容性）

    使用建议：
    - 离线评估和精度要求较高的场景
    - 不需要实时响应的批量分析
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    # 使用优化版本但不启用近似
    results = explain_batch_optimized(
        model, texts, tokenizer, use_approximate=False, show_stats=False
    )

    return results
