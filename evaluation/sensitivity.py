"""
敏感度评估模块

模块功能：
    通过输入扰动评估各可解释性方法的归因稳定性。
    对输入文本施加同义词替换扰动，计算原始归因向量与扰动后归因向量
    之间的余弦相似度，以量化归因结果对输入微小变化的鲁棒性。

余弦相似度作为稳定性指标的合理性：
    余弦相似度衡量两个向量的方向一致性，取值范围 [-1, 1]。
    对于归因分析，语义相似的输入应产生方向一致的归因向量。
    余弦相似度不受向量尺度影响，仅关注归因权重的相对分布，
    适合评估归因模式的稳定性而非绝对数值的一致性。

同义词替换扰动的局限性：
    - 同义词替换可能改变句子的细微语义，导致模型产生合理的归因变化
    - WordNet 的同义词覆盖范围有限，某些专业术语可能无法找到合适的替换词
    - 替换后的句子可能不符合语法规范，引入非预期的归因变化
    - 替换比例和替换词的选取会影响评估结果

依赖模块：
    - explainability/ 各归因模块
    - config.py：超参数配置

作者：Kris
"""

import random
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENSITIVITY_PERTURB_RATIO, SENSITIVITY_NUM_PERTURBATIONS, SEED

# 尝试导入 NLTK WordNet 用于同义词查询
try:
    import nltk
    from nltk.corpus import wordnet
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False


def _ensure_nltk_data():
    """确保 NLTK 的 WordNet 数据已下载"""
    if not WORDNET_AVAILABLE:
        return
    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def _get_synonym(word):
    """
    查询词的同义词

    使用 WordNet 查找给定词的同义词列表，返回与原词不同的第一个同义词。
    若无可用同义词，返回原词。

    参数：
        word: 待查询的词

    返回值：
        str: 同义词或原词
    """
    if not WORDNET_AVAILABLE:
        return word

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)

    if synonyms:
        return random.choice(list(synonyms))
    return word


def perturb_text(text, perturb_ratio=SENSITIVITY_PERTURB_RATIO):
    """
    对文本施加同义词替换扰动

    参数：
        text: 原始文本字符串
        perturb_ratio: 替换词语占总词数的比例

    返回值：
        str: 扰动后的文本
    """
    _ensure_nltk_data()

    words = text.split()
    num_words = len(words)
    num_perturb = max(1, int(num_words * perturb_ratio))

    # 随机选取要替换的词位置
    perturb_indices = random.sample(range(num_words), min(num_perturb, num_words))

    perturbed_words = words.copy()
    for idx in perturb_indices:
        synonym = _get_synonym(words[idx])
        perturbed_words[idx] = synonym

    return " ".join(perturbed_words)


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    参数：
        vec1: 第一个向量（列表或 numpy 数组）
        vec2: 第二个向量（列表或 numpy 数组）

    返回值：
        float: 余弦相似度，取值范围 [-1, 1]
    """
    vec1 = np.array(vec1, dtype=np.float64)
    vec2 = np.array(vec2, dtype=np.float64)

    # 截断至相同长度
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def evaluate_sensitivity(explain_func, texts, method_name, dataset_name,
                         num_perturbations=SENSITIVITY_NUM_PERTURBATIONS):
    """
    对一组样本评估某可解释性方法的敏感度

    参数：
        explain_func: 归因函数，接受文本字符串，返回包含 attribution_scores 的字典
        texts: 文本字符串列表
        method_name: 方法名称（用于结果标记）
        dataset_name: 数据集名称
        num_perturbations: 每条样本生成的扰动版本数量

    返回值：
        dict: 包含以下字段
            - method: 方法名称
            - dataset: 数据集名称
            - mean_sensitivity: 平均余弦相似度（越高表示越稳定）
            - per_sample_scores: 各样本的余弦相似度列表
    """
    random.seed(SEED)

    per_sample_scores = []

    for text in texts:
        try:
            # 获取原始文本的归因结果
            original_result = explain_func(text)
            original_scores = original_result.get("attribution_scores", [])

            if not original_scores:
                continue

            sample_similarities = []

            for _ in range(num_perturbations):
                # 生成扰动文本
                perturbed_text = perturb_text(text)

                # 获取扰动文本的归因结果
                perturbed_result = explain_func(perturbed_text)
                perturbed_scores = perturbed_result.get("attribution_scores", [])

                if not perturbed_scores:
                    continue

                # 计算余弦相似度
                sim = cosine_similarity(original_scores, perturbed_scores)
                sample_similarities.append(sim)

            if sample_similarities:
                # 该样本的敏感度分数为多次扰动的平均余弦相似度
                avg_sim = float(np.mean(sample_similarities))
                per_sample_scores.append(avg_sim)

        except Exception as e:
            print(f"敏感度评估出错: {e}")
            continue

    mean_sensitivity = float(np.mean(per_sample_scores)) if per_sample_scores else 0.0

    return {
        "method": method_name,
        "dataset": dataset_name,
        "mean_sensitivity": mean_sensitivity,
        "per_sample_scores": per_sample_scores,
    }
