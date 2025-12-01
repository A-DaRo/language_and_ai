"""
Text cleaning utilities for NLP preprocessing.
Handles common issues in social media / blog text data.
"""

import re
import unicodedata
from typing import List, Optional, Union
import pandas as pd
import numpy as np


class TextCleaner:
    """
    A comprehensive text cleaning class for NLP preprocessing.
    Designed for social media and blog post data.
    """
    
    # Common URL patterns
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # HTML tag pattern
    HTML_PATTERN = re.compile(r'<[^>]+>')
    
    # Email pattern
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # Username/mention pattern (Twitter-style)
    MENTION_PATTERN = re.compile(r'@\w+')
    
    # Hashtag pattern
    HASHTAG_PATTERN = re.compile(r'#\w+')
    
    # Repeated characters (3+ of same char)
    REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{2,}')
    
    # Multiple spaces
    MULTI_SPACE_PATTERN = re.compile(r'\s+')
    
    # Non-ASCII pattern
    NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]+')
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_emails: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,  # Often contains semantic info
        normalize_unicode: bool = True,
        reduce_repeated_chars: bool = True,
        min_token_length: int = 1,
        max_token_length: int = 50
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.normalize_unicode = normalize_unicode
        self.reduce_repeated_chars = reduce_repeated_chars
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps to a single text."""
        if not isinstance(text, str):
            return ""
        
        # Handle empty or whitespace-only strings
        if not text.strip():
            return ""
        
        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
            text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove HTML tags
        if self.remove_html:
            text = self.HTML_PATTERN.sub(' ', text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)
        
        # Remove emails
        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self.MENTION_PATTERN.sub(' ', text)
        
        # Remove hashtags (but keep the text without #)
        if self.remove_hashtags:
            text = self.HASHTAG_PATTERN.sub(' ', text)
        else:
            # Keep hashtag content, remove # symbol
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Reduce repeated characters
        if self.reduce_repeated_chars:
            text = self.REPEATED_CHARS_PATTERN.sub(r'\1\1', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = self.MULTI_SPACE_PATTERN.sub(' ', text)
        
        return text.strip()
    
    def clean_series(self, series: pd.Series, show_progress: bool = False) -> pd.Series:
        """Clean a pandas Series of texts."""
        if show_progress:
            try:
                from tqdm import tqdm
                tqdm.pandas()
                return series.progress_apply(self.clean_text)
            except ImportError:
                pass
        return series.apply(self.clean_text)
    
    def get_cleaning_stats(self, original: pd.Series, cleaned: pd.Series) -> dict:
        """Compare original vs cleaned text to quantify cleaning impact."""
        orig_lens = original.astype(str).str.len()
        clean_lens = cleaned.astype(str).str.len()
        
        return {
            'original_mean_length': orig_lens.mean(),
            'cleaned_mean_length': clean_lens.mean(),
            'length_reduction_pct': 100 * (1 - clean_lens.sum() / orig_lens.sum()),
            'empty_after_cleaning': (clean_lens == 0).sum(),
            'original_empty': (orig_lens == 0).sum(),
            'total_chars_removed': orig_lens.sum() - clean_lens.sum()
        }


def detect_language_heuristic(text: str) -> str:
    """
    Simple heuristic language detection based on character patterns.
    Returns 'en' for English-like, 'other' for non-English.
    """
    if not isinstance(text, str) or len(text) < 10:
        return 'unknown'
    
    # Count ASCII vs non-ASCII characters
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total_chars = len(text)
    
    ascii_ratio = ascii_chars / total_chars if total_chars > 0 else 0
    
    # High ASCII ratio suggests English/Western language
    if ascii_ratio > 0.9:
        return 'en'
    elif ascii_ratio > 0.7:
        return 'mixed'
    else:
        return 'other'


def flatten_list_column(series: pd.Series) -> pd.Series:
    """
    Handle columns that may contain lists (from merging operations).
    Joins list elements with space separator.
    """
    def flatten_item(x):
        # Handle NaN/None first
        if pd.isna(x):
            return ''
        # Handle list/tuple/set
        if isinstance(x, (list, tuple, set)):
            # Recursively flatten nested structures
            flattened = []
            for item in x:
                if isinstance(item, (list, tuple, set)):
                    flattened.extend(str(i) for i in item if pd.notna(i))
                elif pd.notna(item):
                    flattened.append(str(item))
            return ' '.join(flattened)
        # Handle scalar values
        return str(x)
    
    return series.apply(flatten_item)
