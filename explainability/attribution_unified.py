"""
统一的归因分数处理与可视化模块

模块功能：
    为不同的可解释性方法的归因结果提供统一的数据格式、归一化处理和可视化标准。

    不同方法的归因分数在形式和尺度上存在差异：
    - 注意力可视化：直接输出概率分布，范围 [0, 1]
    - Grad-CAM：梯度加权激活，范围可能较大
    - Integrated Gradients：积分梯度，范围 [-inf, +inf]
    - SHAP：Shapley 值，通常在 [-1, 1] 范围
    - LIME：局部线性系数，范围不固定

    统一处理包括：
    1. 归一化：将所有分数映射到 [0, 1] 区间
    2. 标准化：计算 z-score 进行标准差标准化
    3. 排序：支持按分数大小排序
    4. 统计：提供分数的统计信息（均值、方差、极值等）

依赖模块：
    无额外依赖，仅使用 numpy

作者：Kris
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """
    统一的归因结果数据类

    属性：
        tokens: token 字符串列表
        method: 归因方法名称 (str: "attention", "gradcam", "ig", "shap", "lime")
        raw_scores: 原始归因分数
        normalized_scores: 归一化到 [0, 1] 的分数
        standardized_scores: 标准化的分数 (z-score)
        token_importance: 按重要性排序的 token 列表
        statistics: 分数统计信息
    """

    tokens: List[str]
    method: str
    raw_scores: List[float]
    normalized_scores: List[float]
    standardized_scores: List[float]
    token_importance: List[Dict]  # [{"token": str, "score": float, "rank": int}, ...]
    statistics: Dict[
        str, float
    ]  # {"mean": float, "std": float, "min": float, "max": float, ...}

    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            "tokens": self.tokens,
            "method": self.method,
            "raw_scores": self.raw_scores,
            "normalized_scores": self.normalized_scores,
            "standardized_scores": self.standardized_scores,
            "token_importance": self.token_importance,
            "statistics": self.statistics,
        }


class AttributionNormalizer:
    """
    归因分数处理器，提供多种归一化和标准化方法
    """

    @staticmethod
    def normalize_minmax(scores: List[float]) -> List[float]:
        """
        Min-Max 归一化：将分数映射到 [0, 1] 区间

        公式：x_norm = (x - min) / (max - min)

        参数：
            scores: 原始分数列表

        返回值：
            归一化后的分数列表，范围 [0, 1]
        """
        scores_array = np.array(scores, dtype=float)

        # 处理常数列（所有值相同）
        if np.max(scores_array) == np.min(scores_array):
            return np.zeros_like(scores_array).tolist()

        normalized = (scores_array - np.min(scores_array)) / (
            np.max(scores_array) - np.min(scores_array)
        )
        return normalized.tolist()

    @staticmethod
    def normalize_zscore(scores: List[float]) -> List[float]:
        """
        Z-score 标准化：(x - mean) / std

        用于比较来自不同分布的分数，突出异常值

        参数：
            scores: 原始分数列表

        返回值：
            标准化后的分数列表，均值为 0，标准差为 1
        """
        scores_array = np.array(scores, dtype=float)
        mean = np.mean(scores_array)
        std = np.std(scores_array)

        if std == 0:
            return np.zeros_like(scores_array).tolist()

        standardized = (scores_array - mean) / std
        return standardized.tolist()

    @staticmethod
    def normalize_softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
        """
        Softmax 归一化：exp(x/T) / sum(exp(x/T))

        适合将分数转换为概率分布，temperature 参数控制集中度

        参数：
            scores: 原始分数列表
            temperature: 温度参数，越小越集中

        返回值：
            概率分布，和为 1
        """
        scores_array = np.array(scores, dtype=float)

        # 防止数值溢出
        scores_shifted = scores_array - np.max(scores_array)
        exp_scores = np.exp(scores_shifted / temperature)
        softmax_scores = exp_scores / np.sum(exp_scores)

        return softmax_scores.tolist()

    @staticmethod
    def compute_statistics(scores: List[float]) -> Dict[str, float]:
        """
        计算分数的统计信息

        参数：
            scores: 分数列表

        返回值：
            包含以下字段的字典：
            - mean: 均值
            - std: 标准差
            - min: 最小值
            - max: 最大值
            - median: 中位数
            - q25: 25 分位数
            - q75: 75 分位数
            - range: 极差 (max - min)
        """
        scores_array = np.array(scores, dtype=float)

        return {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
            "q25": float(np.percentile(scores_array, 25)),
            "q75": float(np.percentile(scores_array, 75)),
            "range": float(np.max(scores_array) - np.min(scores_array)),
        }


def unify_attribution_result(
    tokens: List[str],
    method: str,
    scores: List[float],
    normalize_method: str = "minmax",
) -> AttributionResult:
    """
    统一处理单个方法的归因结果

    参数：
        tokens: token 字符串列表
        method: 归因方法名称 ("attention", "gradcam", "ig", "shap", "lime")
        scores: 原始归因分数
        normalize_method: 归一化方法 ("minmax", "zscore", "softmax")

    返回值：
        AttributionResult 对象，包含统一格式的结果
    """
    # 截断至 token 数量
    scores = scores[: len(tokens)]

    # 选择归一化方法
    if normalize_method == "minmax":
        normalized = AttributionNormalizer.normalize_minmax(scores)
    elif normalize_method == "zscore":
        normalized = AttributionNormalizer.normalize_zscore(scores)
    elif normalize_method == "softmax":
        normalized = AttributionNormalizer.normalize_softmax(scores)
    else:
        normalized = AttributionNormalizer.normalize_minmax(scores)

    # 计算 z-score 标准化（用于统计比较）
    standardized = AttributionNormalizer.normalize_zscore(scores)

    # 计算统计信息
    statistics = AttributionNormalizer.compute_statistics(scores)

    # 生成按重要性排序的 token 列表
    token_importance = []
    for i, (token, score) in enumerate(zip(tokens, normalized)):
        if token not in ["[PAD]", "[CLS]", "[SEP]"]:  # 排除特殊 token
            token_importance.append(
                {
                    "token": token,
                    "raw_score": float(scores[i]),
                    "normalized_score": float(score),
                    "position": i,
                }
            )

    # 按归一化分数排序
    token_importance.sort(key=lambda x: x["normalized_score"], reverse=True)

    # 添加排名
    for rank, item in enumerate(token_importance, 1):
        item["rank"] = rank

    return AttributionResult(
        tokens=tokens,
        method=method,
        raw_scores=scores,
        normalized_scores=normalized,
        standardized_scores=standardized,
        token_importance=token_importance,
        statistics=statistics,
    )


def unify_multiple_methods(
    results_dict: Dict[str, Dict],
    normalize_method: str = "minmax",
) -> Dict[str, AttributionResult]:
    """
    批量统一处理多个方法的归因结果

    参数：
        results_dict: 方法名 -> 原始结果的字典
                      每个结果应包含 "tokens" 和 "scores" 字段
        normalize_method: 归一化方法

    返回值：
        方法名 -> AttributionResult 的字典

    示例：
        results = {
            "attention": {"tokens": [...], "scores": [...]},
            "gradcam": {"tokens": [...], "scores": [...]},
            "ig": {"tokens": [...], "scores": [...]},
        }
        unified = unify_multiple_methods(results)
    """
    unified_results = {}

    for method, result in results_dict.items():
        unified_results[method] = unify_attribution_result(
            tokens=result.get("tokens", []),
            method=method,
            scores=result.get("scores", []),
            normalize_method=normalize_method,
        )

    return unified_results


class AttributionComparator:
    """
    用于比较多个方法的归因结果
    """

    @staticmethod
    def get_consensus_important_tokens(
        results: Dict[str, AttributionResult],
        top_k: int = 10,
        consensus_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        获取多个方法共识认为重要的 token

        参数：
            results: 方法名 -> AttributionResult 的字典
            top_k: 每个方法取前 k 个重要 token
            consensus_threshold: 共识阈值，0.5 表示至少 50% 的方法认为重要

        返回值：
            重要 token 列表，按共识强度排序
        """
        # 统计各 token 在方法中被认为重要的次数
        token_consensus = {}

        for method, result in results.items():
            for item in result.token_importance[:top_k]:
                token = item["token"]
                if token not in token_consensus:
                    token_consensus[token] = {
                        "count": 0,
                        "positions": [],
                        "scores": {},
                    }
                token_consensus[token]["count"] += 1
                token_consensus[token]["positions"].append(item["position"])
                token_consensus[token]["scores"][method] = item["normalized_score"]

        # 过滤达到共识阈值的 token
        num_methods = len(results)
        threshold_count = int(consensus_threshold * num_methods)

        consensus_tokens = []
        for token, info in token_consensus.items():
            if info["count"] >= threshold_count:
                consensus_tokens.append(
                    {
                        "token": token,
                        "consensus_strength": info["count"] / num_methods,
                        "average_position": float(np.mean(info["positions"])),
                        "average_score": float(np.mean(list(info["scores"].values()))),
                        "method_scores": info["scores"],
                    }
                )

        # 按共识强度排序
        consensus_tokens.sort(key=lambda x: x["consensus_strength"], reverse=True)

        return consensus_tokens

    @staticmethod
    def compute_method_correlation(
        results: Dict[str, AttributionResult],
    ) -> Dict[Tuple[str, str], float]:
        """
        计算方法间的相关性（Spearman 相关系数）

        参数：
            results: 方法名 -> AttributionResult 的字典

        返回值：
            (method1, method2) -> 相关系数的字典
        """
        from scipy.stats import spearmanr

        correlations = {}
        method_names = list(results.keys())

        for i, method1 in enumerate(method_names):
            for method2 in method_names[i + 1 :]:
                scores1 = results[method1].normalized_scores
                scores2 = results[method2].normalized_scores

                corr, _ = spearmanr(scores1, scores2)
                correlations[(method1, method2)] = float(corr)

        return correlations


def format_for_visualization(
    unified_results: Dict[str, AttributionResult],
) -> Dict[str, Dict]:
    """
    将统一格式的结果转换为可视化友好的格式

    参数：
        unified_results: 方法名 -> AttributionResult 的字典

    返回值：
        包含以下内容的字典：
        - "tokens": 统一的 token 列表
        - "methods": {方法名 -> 可视化数据}
        - "heatmap_data": 热力图数据矩阵
        - "colorscale": 推荐的颜色方案
    """
    # 使用第一个方法的 token 作为基准
    first_method = list(unified_results.keys())[0]
    tokens = unified_results[first_method].tokens

    viz_data = {
        "tokens": tokens,
        "methods": {},
        "heatmap_data": [],
        "colorscale": "RdYlGn",  # 热力图颜色方案：红-黄-绿
    }

    for method, result in unified_results.items():
        viz_data["methods"][method] = {
            "normalized_scores": result.normalized_scores,
            "standardized_scores": result.standardized_scores,
            "token_importance": result.token_importance[:10],  # Top-10
            "statistics": result.statistics,
        }
        viz_data["heatmap_data"].append(result.normalized_scores)

    return viz_data
