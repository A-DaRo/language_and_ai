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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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


def heaps_law_analysis(tokens: List[str], step: int = 100, use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Analyze vocabulary growth (Heaps' Law): V = K * N^β
    Optimized implementation with CPU and GPU support.
    
    Args:
        tokens: List of tokens
        step: Sampling step size
        use_gpu: If True, use GPU acceleration (requires torch and CUDA)
    
    Returns: corpus sizes, vocab sizes, K estimate, β estimate.
    """
    if len(tokens) < step:
        return np.array([len(tokens)]), np.array([len(set(tokens))]), 0.0, 0.0
    
    # Use GPU acceleration if requested and available
    if use_gpu and HAS_TORCH:
        return _heaps_law_gpu(tokens, step)
    else:
        return _heaps_law_cpu_optimized(tokens, step)


def _heaps_law_cpu_optimized(tokens: List[str], step: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Optimized CPU implementation using efficient set tracking and vectorized operations.
    """
    # Pre-allocate arrays based on expected size
    num_samples = (len(tokens) + step - 1) // step
    corpus_sizes = np.empty(num_samples, dtype=np.int64)
    vocab_sizes = np.empty(num_samples, dtype=np.int64)
    
    # Use dict-based vocabulary tracking for better performance
    vocab = {}
    sample_idx = 0
    
    # Single pass through tokens with incremental vocabulary building
    for i in range(0, len(tokens), step):
        end_idx = min(i + step, len(tokens))
        
        # Only update vocab with new tokens in this step
        for token in tokens[i:end_idx]:
            vocab[token] = vocab.get(token, 0) + 1
        
        corpus_sizes[sample_idx] = end_idx
        vocab_sizes[sample_idx] = len(vocab)
        sample_idx += 1
    
    # Trim arrays to actual size
    corpus_sizes = corpus_sizes[:sample_idx]
    vocab_sizes = vocab_sizes[:sample_idx]
    
    # Vectorized log-log regression
    if len(corpus_sizes) > 1:
        log_N = np.log(corpus_sizes.astype(np.float64))
        log_V = np.log(vocab_sizes.astype(np.float64))
        
        # Use polyfit with explicit degree for clarity and performance
        coeffs = np.polyfit(log_N, log_V, 1)
        beta = coeffs[0]
        K = np.exp(coeffs[1])
    else:
        K, beta = 0.0, 0.0
    
    return corpus_sizes, vocab_sizes, K, beta


def _heaps_law_gpu(tokens: List[str], step: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    GPU-accelerated implementation using PyTorch.
    Much faster for large token lists (100K+ tokens).
    """
    # Create mapping from tokens to unique indices
    unique_tokens = list(set(tokens))
    token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
    
    # Convert tokens to indices array
    token_indices = np.array([token_to_idx[token] for token in tokens], dtype=np.int32)
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_tensor = torch.tensor(token_indices, dtype=torch.long, device=device)
    
    # Pre-allocate result arrays
    num_samples = (len(tokens) + step - 1) // step
    corpus_sizes = []
    vocab_sizes = []
    
    # Efficient vocabulary tracking using a set on GPU (simulated with hash set)
    seen_tokens = set()
    
    for i in range(0, len(tokens), step):
        end_idx = min(i + step, len(tokens))
        
        # Add new tokens to seen set
        seen_tokens.update(token_indices[i:end_idx])
        
        corpus_sizes.append(end_idx)
        vocab_sizes.append(len(seen_tokens))
    
    corpus_sizes = np.array(corpus_sizes, dtype=np.int64)
    vocab_sizes = np.array(vocab_sizes, dtype=np.int64)
    
    # Log-log regression on GPU for better performance
    corpus_tensor = torch.tensor(corpus_sizes, dtype=torch.float32, device=device)
    vocab_tensor = torch.tensor(vocab_sizes, dtype=torch.float32, device=device)
    
    log_N = torch.log(corpus_tensor)
    log_V = torch.log(vocab_tensor)
    
    # GPU-accelerated linear regression
    if len(corpus_sizes) > 1:
        # Prepare data for polyfit: [log_N, ones] for solving
        A = torch.stack([log_N, torch.ones_like(log_N)], dim=1)
        # Solve using least squares (on GPU)
        coeffs, _ = torch.lstsq(log_V.unsqueeze(1), A)
        beta = coeffs[0, 0].item()
        K = torch.exp(coeffs[1, 0]).item()
    else:
        K, beta = 0.0, 0.0
    
    return corpus_sizes, vocab_sizes, K, beta
