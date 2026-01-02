"""
NLP metrics for deeper text analysis.
Includes embedding-based metrics, entropy measures, and distributional semantics.
"""

import math
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import numpy as np
import pandas as pd


class NLPMetrics:
    """
    Advanced NLP metrics including entropy, perplexity estimates,
    and vocabulary overlap measures.
    """
    
    @staticmethod
    def shannon_entropy(tokens: List[str]) -> float:
        """
        Shannon entropy of token distribution.
        H = -sum(p_i * log2(p_i))
        Higher values indicate more uniform distribution.
        """
        if not tokens:
            return 0.0
        
        freq = Counter(tokens)
        total = len(tokens)
        
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    @staticmethod
    def normalized_entropy(tokens: List[str]) -> float:
        """
        Normalized entropy (0-1 scale).
        H_norm = H / log2(vocab_size)
        """
        if not tokens:
            return 0.0
        
        vocab_size = len(set(tokens))
        if vocab_size <= 1:
            return 0.0
        
        H = NLPMetrics.shannon_entropy(tokens)
        return H / math.log2(vocab_size)
    
    @staticmethod
    def conditional_entropy_bigrams(tokens: List[str]) -> float:
        """
        Conditional entropy H(w_i | w_{i-1}) using bigrams.
        Measures predictability of next word given previous.
        """
        if len(tokens) < 2:
            return 0.0
        
        # Count bigrams and unigrams
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        bigram_freq = Counter(bigrams)
        unigram_freq = Counter(tokens[:-1])
        
        H = 0.0
        for bigram, count in bigram_freq.items():
            p_bigram = count / len(bigrams)
            p_conditional = count / unigram_freq[bigram[0]]
            if p_conditional > 0:
                H -= p_bigram * math.log2(p_conditional)
        
        return H
    
    @staticmethod
    def perplexity_unigram(tokens: List[str]) -> float:
        """
        Unigram perplexity estimate.
        PP = 2^H where H is entropy.
        Lower perplexity indicates more predictable text.
        """
        H = NLPMetrics.shannon_entropy(tokens)
        return 2 ** H if H > 0 else 1.0
    
    @staticmethod
    def vocabulary_overlap(tokens1: List[str], tokens2: List[str]) -> Dict[str, float]:
        """
        Compute vocabulary overlap metrics between two token lists.
        """
        vocab1 = set(tokens1)
        vocab2 = set(tokens2)
        
        intersection = vocab1 & vocab2
        union = vocab1 | vocab2
        
        return {
            'jaccard_similarity': len(intersection) / len(union) if union else 0.0,
            'dice_coefficient': 2 * len(intersection) / (len(vocab1) + len(vocab2)) if (vocab1 or vocab2) else 0.0,
            'overlap_coefficient': len(intersection) / min(len(vocab1), len(vocab2)) if (vocab1 and vocab2) else 0.0,
            'vocab1_coverage': len(intersection) / len(vocab1) if vocab1 else 0.0,
            'vocab2_coverage': len(intersection) / len(vocab2) if vocab2 else 0.0,
            'unique_to_vocab1': len(vocab1 - vocab2),
            'unique_to_vocab2': len(vocab2 - vocab1),
            'shared_vocab': len(intersection)
        }
    
    @staticmethod
    def cosine_similarity_bow(tokens1: List[str], tokens2: List[str]) -> float:
        """
        Cosine similarity using bag-of-words representation.
        """
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        
        vocab = set(freq1.keys()) | set(freq2.keys())
        
        dot_product = sum(freq1.get(w, 0) * freq2.get(w, 0) for w in vocab)
        norm1 = math.sqrt(sum(v ** 2 for v in freq1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in freq2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def lexical_density(tokens: List[str], content_words: Optional[Set[str]] = None) -> float:
        """
        Lexical density: ratio of content words to total words.
        Higher density indicates more information-packed text.
        
        If content_words not provided, uses heuristic based on word length.
        """
        if not tokens:
            return 0.0
        
        if content_words is None:
            # Heuristic: words >= 4 chars are likely content words
            content_count = sum(1 for t in tokens if len(t) >= 4)
        else:
            content_count = sum(1 for t in tokens if t in content_words)
        
        return content_count / len(tokens)
    
    @staticmethod
    def burstiness(tokens: List[str], target_word: str) -> float:
        """
        Burstiness of a word: measures clustering tendency.
        B = (σ - μ) / (σ + μ) where σ is std and μ is mean of inter-arrival times.
        B = 1: highly bursty, B = 0: random, B = -1: periodic.
        """
        positions = [i for i, t in enumerate(tokens) if t == target_word]
        
        if len(positions) < 2:
            return 0.0
        
        # Inter-arrival times
        gaps = np.diff(positions)
        
        mu = np.mean(gaps)
        sigma = np.std(gaps)
        
        if mu + sigma == 0:
            return 0.0
        
        return (sigma - mu) / (sigma + mu)
    
    @staticmethod
    def average_burstiness(tokens: List[str], min_freq: int = 5) -> float:
        """
        Average burstiness across all words with minimum frequency.
        """
        freq = Counter(tokens)
        frequent_words = [w for w, c in freq.items() if c >= min_freq]
        
        if not frequent_words:
            return 0.0
        
        burstiness_values = [NLPMetrics.burstiness(tokens, w) for w in frequent_words]
        return np.mean(burstiness_values)
    
    @staticmethod
    def word_frequency_distribution(tokens: List[str]) -> Dict[str, float]:
        """
        Analyze word frequency distribution statistics.
        """
        if not tokens:
            return {}
        
        freq = Counter(tokens)
        frequencies = list(freq.values())
        
        return {
            'mean_frequency': np.mean(frequencies),
            'median_frequency': np.median(frequencies),
            'max_frequency': max(frequencies),
            'frequency_std': np.std(frequencies),
            'frequency_skewness': NLPMetrics._skewness(frequencies),
            'frequency_kurtosis': NLPMetrics._kurtosis(frequencies),
            'hapax_count': sum(1 for f in frequencies if f == 1),
            'dis_legomena_count': sum(1 for f in frequencies if f == 2),
            'high_freq_words': sum(1 for f in frequencies if f >= 10)
        }
    
    @staticmethod
    def _skewness(data: List[float]) -> float:
        """Compute skewness of a distribution."""
        if len(data) < 3:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std) ** 3 for x in data)
    
    @staticmethod
    def _kurtosis(data: List[float]) -> float:
        """Compute excess kurtosis of a distribution."""
        if len(data) < 4:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * \
               sum(((x - mean) / std) ** 4 for x in data)
        adjustment = 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
        return kurt - adjustment
    
    @staticmethod
    def pointwise_mutual_information(tokens: List[str], word1: str, word2: str, 
                                     window_size: int = 5) -> float:
        """
        Pointwise Mutual Information between two words.
        PMI(x,y) = log2(P(x,y) / (P(x) * P(y)))
        Positive PMI indicates words co-occur more than by chance.
        """
        freq = Counter(tokens)
        n = len(tokens)
        
        if word1 not in freq or word2 not in freq:
            return 0.0
        
        p_x = freq[word1] / n
        p_y = freq[word2] / n
        
        # Count co-occurrences within window
        cooccurrences = 0
        total_windows = 0
        
        for i in range(len(tokens)):
            window_start = max(0, i - window_size)
            window_end = min(len(tokens), i + window_size + 1)
            window = tokens[window_start:window_end]
            
            if tokens[i] == word1:
                cooccurrences += window.count(word2) - (1 if word1 == word2 else 0)
                total_windows += len(window) - 1
        
        if total_windows == 0 or cooccurrences == 0:
            return 0.0
        
        p_xy = cooccurrences / total_windows
        
        if p_x * p_y == 0:
            return 0.0
        
        return math.log2(p_xy / (p_x * p_y))


def compute_ngram_statistics(tokens: List[str], n: int = 2) -> Dict[str, float]:
    """
    Compute statistics for n-grams.
    """
    if len(tokens) < n:
        return {'ngram_count': 0, 'unique_ngrams': 0, 'ngram_ttr': 0.0}
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    freq = Counter(ngrams)
    
    return {
        'ngram_count': len(ngrams),
        'unique_ngrams': len(freq),
        'ngram_ttr': len(freq) / len(ngrams) if ngrams else 0.0,
        'ngram_hapax_ratio': sum(1 for c in freq.values() if c == 1) / len(ngrams) if ngrams else 0.0,
        'ngram_entropy': NLPMetrics.shannon_entropy([str(ng) for ng in ngrams]),
        'most_common_ngram_freq': freq.most_common(1)[0][1] / len(ngrams) if freq else 0.0
    }


def vocabulary_richness_summary(tokens: List[str]) -> Dict[str, float]:
    """
    Comprehensive vocabulary richness summary.
    """
    from .text_stats import TextStatistics
    
    if not tokens:
        return {}
    
    ts = TextStatistics()
    
    return {
        'vocabulary_size': len(set(tokens)),
        'token_count': len(tokens),
        'ttr': ts.type_token_ratio(tokens),
        'root_ttr': ts.root_ttr(tokens),
        'log_ttr': ts.log_ttr(tokens),
        'mattr': ts.moving_average_ttr(tokens),
        'hapax_ratio': ts.hapax_legomena_ratio(tokens),
        'yules_k': ts.yules_k(tokens),
        'simpsons_d': ts.simpsons_d(tokens),
        'entropy': NLPMetrics.shannon_entropy(tokens),
        'normalized_entropy': NLPMetrics.normalized_entropy(tokens)
    }
