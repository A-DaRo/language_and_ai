"""
Text statistics utilities for analyzing linguistic properties.
Includes lexical diversity, readability metrics, and distributional statistics.
"""

import re
import math
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import numpy as np
import pandas as pd


class TextStatistics:
    """
    Compute various text statistics for NLP analysis.
    Focuses on lexical, syntactic, and distributional properties.
    """
    
    # Simple sentence boundary pattern
    SENTENCE_PATTERN = re.compile(r'[.!?]+')
    
    # Word tokenization pattern (simple)
    WORD_PATTERN = re.compile(r'\b[a-zA-Z]+\b')
    
    def __init__(self, min_word_length: int = 1):
        self.min_word_length = min_word_length
    
    @staticmethod
    def tokenize_simple(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        if not isinstance(text, str):
            return []
        return TextStatistics.WORD_PATTERN.findall(text.lower())
    
    @staticmethod
    def count_sentences(text: str) -> int:
        """Estimate sentence count."""
        if not isinstance(text, str) or not text.strip():
            return 0
        sentences = TextStatistics.SENTENCE_PATTERN.split(text)
        return len([s for s in sentences if s.strip()])
    
    @staticmethod
    def type_token_ratio(tokens: List[str]) -> float:
        """
        Type-Token Ratio (TTR): measure of lexical diversity.
        TTR = unique_words / total_words
        Higher values indicate more diverse vocabulary.
        """
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)
    
    @staticmethod
    def root_ttr(tokens: List[str]) -> float:
        """
        Root TTR (Guiraud's R): corrects for text length bias.
        R = unique_words / sqrt(total_words)
        """
        if not tokens:
            return 0.0
        return len(set(tokens)) / math.sqrt(len(tokens))
    
    @staticmethod
    def log_ttr(tokens: List[str]) -> float:
        """
        Log TTR (Herdan's C): another length-corrected diversity measure.
        C = log(unique_words) / log(total_words)
        """
        if len(tokens) <= 1:
            return 0.0
        unique = len(set(tokens))
        if unique <= 1:
            return 0.0
        return math.log(unique) / math.log(len(tokens))
    
    @staticmethod
    def moving_average_ttr(tokens: List[str], window_size: int = 100) -> float:
        """
        Moving-Average TTR (MATTR): robust to text length.
        Computes TTR in sliding windows and averages.
        """
        if len(tokens) < window_size:
            return TextStatistics.type_token_ratio(tokens)
        
        ttrs = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            ttrs.append(len(set(window)) / window_size)
        
        return np.mean(ttrs)
    
    @staticmethod
    def hapax_legomena_ratio(tokens: List[str]) -> float:
        """
        Ratio of words appearing exactly once (hapax legomena).
        High ratio indicates rich vocabulary usage.
        """
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        hapax = sum(1 for count in freq.values() if count == 1)
        return hapax / len(tokens)
    
    @staticmethod
    def yules_k(tokens: List[str]) -> float:
        """
        Yule's K characteristic: vocabulary richness measure.
        Lower values indicate richer vocabulary.
        Formula: K = 10^4 * (sum(freq_i^2 * i^2) - N) / N^2
        """
        if len(tokens) < 2:
            return 0.0
        
        freq = Counter(tokens)
        N = len(tokens)
        
        # Count frequency of frequencies
        freq_of_freq = Counter(freq.values())
        
        M1 = N
        M2 = sum(count * (i ** 2) for i, count in freq_of_freq.items())
        
        if M1 == 0:
            return 0.0
        
        K = 10000 * (M2 - M1) / (M1 ** 2)
        return K
    
    @staticmethod
    def simpsons_d(tokens: List[str]) -> float:
        """
        Simpson's Diversity Index: probability two random words are different.
        Higher values indicate more diversity.
        """
        if len(tokens) < 2:
            return 0.0
        
        freq = Counter(tokens)
        N = len(tokens)
        
        D = 1 - sum(n * (n - 1) for n in freq.values()) / (N * (N - 1))
        return D
    
    @staticmethod
    def average_word_length(tokens: List[str]) -> float:
        """Average word length in characters."""
        if not tokens:
            return 0.0
        return np.mean([len(t) for t in tokens])
    
    @staticmethod
    def word_length_variance(tokens: List[str]) -> float:
        """Variance in word lengths."""
        if len(tokens) < 2:
            return 0.0
        return np.var([len(t) for t in tokens])
    
    @staticmethod
    def flesch_reading_ease(text: str) -> float:
        """
        Flesch Reading Ease score.
        Higher = easier to read (60-70 is standard, 30-50 is college level).
        """
        tokens = TextStatistics.tokenize_simple(text)
        sentences = TextStatistics.count_sentences(text)
        
        if not tokens or sentences == 0:
            return 0.0
        
        # Estimate syllables (simple heuristic)
        syllables = sum(TextStatistics._count_syllables(word) for word in tokens)
        
        words = len(tokens)
        
        # Flesch formula
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0, min(100, score))
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count using simple vowel-based heuristic."""
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    @staticmethod
    def automated_readability_index(text: str) -> float:
        """
        Automated Readability Index (ARI).
        Estimates US grade level needed to comprehend the text.
        """
        tokens = TextStatistics.tokenize_simple(text)
        sentences = TextStatistics.count_sentences(text)
        
        if not tokens or sentences == 0:
            return 0.0
        
        chars = sum(len(t) for t in tokens)
        words = len(tokens)
        
        ARI = 4.71 * (chars / words) + 0.5 * (words / sentences) - 21.43
        return max(0, ARI)
    
    def compute_all_stats(self, text: str) -> Dict[str, float]:
        """Compute all available statistics for a text."""
        tokens = self.tokenize_simple(text)
        
        # Filter by min word length
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        
        return {
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'sentence_count': self.count_sentences(text),
            'char_count': len(text) if isinstance(text, str) else 0,
            'ttr': self.type_token_ratio(tokens),
            'root_ttr': self.root_ttr(tokens),
            'log_ttr': self.log_ttr(tokens),
            'mattr_100': self.moving_average_ttr(tokens, 100),
            'hapax_ratio': self.hapax_legomena_ratio(tokens),
            'yules_k': self.yules_k(tokens),
            'simpsons_d': self.simpsons_d(tokens),
            'avg_word_length': self.average_word_length(tokens),
            'word_length_var': self.word_length_variance(tokens),
            'flesch_reading_ease': self.flesch_reading_ease(text),
            'ari': self.automated_readability_index(text),
            'avg_sentence_length': len(tokens) / max(1, self.count_sentences(text))
        }
    
    def compute_corpus_stats(self, texts: List[str]) -> pd.DataFrame:
        """Compute statistics for a corpus of texts."""
        results = []
        for text in texts:
            results.append(self.compute_all_stats(text))
        return pd.DataFrame(results)


def zipfs_law_analysis(tokens: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Analyze adherence to Zipf's Law.
    Returns ranks, frequencies, and Zipf exponent estimate.
    """
    if not tokens:
        return np.array([]), np.array([]), 0.0
    
    freq = Counter(tokens)
    sorted_freq = sorted(freq.values(), reverse=True)
    
    ranks = np.arange(1, len(sorted_freq) + 1)
    frequencies = np.array(sorted_freq)
    
    # Estimate Zipf exponent via log-log linear regression
    if len(ranks) > 1:
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)
        
        # Simple linear regression
        slope, _ = np.polyfit(log_ranks, log_freqs, 1)
        zipf_exponent = -slope
    else:
        zipf_exponent = 0.0
    
    return ranks, frequencies, zipf_exponent


def heaps_law_analysis(tokens: List[str], step: int = 100) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Analyze vocabulary growth (Heaps' Law): V = K * N^β
    Returns: corpus sizes, vocab sizes, K estimate, β estimate.
    """
    if len(tokens) < step:
        return np.array([len(tokens)]), np.array([len(set(tokens))]), 0.0, 0.0
    
    corpus_sizes = []
    vocab_sizes = []
    
    seen = set()
    for i in range(0, len(tokens), step):
        seen.update(tokens[:i + step])
        corpus_sizes.append(min(i + step, len(tokens)))
        vocab_sizes.append(len(seen))
    
    corpus_sizes = np.array(corpus_sizes)
    vocab_sizes = np.array(vocab_sizes)
    
    # Estimate Heaps' parameters via log-log regression
    if len(corpus_sizes) > 1:
        log_N = np.log(corpus_sizes)
        log_V = np.log(vocab_sizes)
        
        beta, log_K = np.polyfit(log_N, log_V, 1)
        K = np.exp(log_K)
    else:
        K, beta = 0.0, 0.0
    
    return corpus_sizes, vocab_sizes, K, beta
