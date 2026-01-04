"""
Span Masker for Pollution Mitigation.

Applies typed masks to detected pollution spans in a token-aware manner.

Implements: FR-08 (Safe Masking), GLiNER_Implementation_Strategy.md Section 2.3
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MaskingResult:
    """
    Result of masking operation.
    
    Attributes:
        masked_text: Text with typed masks applied.
        mask_log: Log entries for POLLUTION_LOG_SCHEMA.
    """
    masked_text: str
    mask_log: List[Dict[str, Any]]


class SpanMasker:
    """
    Token-aware span masker with typed mask tokens.
    
    Features:
    - Maps entity types to typed mask tokens
    - Applies masks in reverse order to preserve character offsets
    - Validates single-token encoding of masks
    - Emits structured mask logs for POLLUTION_LOG_SCHEMA
    
    Implements: FR-08
    """
    
    # Default entity type -> Typed mask token mapping
    ENTITY_TO_MASK = {
        "age_statement": "[MASK:AGE]",
        "birth_year_statement": "[MASK:BIRTH_YEAR]",
        "gender_indicator": "[MASK:GENDER]",
        "nationality_statement": "[MASK:NATIONALITY]",
        "country_of_origin": "[MASK:COUNTRY]",
        "demonym": "[MASK:DEMONYM]",
        "personality_type_identifier": "[MASK:PERSONALITY]",
        "mbti_type": "[MASK:MBTI]",
        "political_affiliation": "[MASK:POLITICAL]",
        "ideology_self_id": "[MASK:IDEOLOGY]",
    }
    
    def __init__(
        self,
        tokenizer=None,
        entity_to_mask: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize masker.
        
        Args:
            tokenizer: Optional HuggingFace tokenizer for validation.
        """
        self.tokenizer = tokenizer
        self.entity_to_mask = entity_to_mask or dict(self.ENTITY_TO_MASK)
        
        # Validate single-token encoding if tokenizer provided
        if self.tokenizer:
            self._validate_mask_tokens()
    
    def _validate_mask_tokens(self) -> None:
        """Validate that mask tokens are encoded as single tokens."""
        for entity_type, mask_token in self.entity_to_mask.items():
            token_ids = self.tokenizer.encode(mask_token, add_special_tokens=False)
            if len(token_ids) != 1:
                logger.warning(
                    f"Mask token {mask_token} for {entity_type} "
                    f"encoded as {len(token_ids)} tokens: {token_ids}"
                )
    
    def mask_spans(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        post_id: str,
    ) -> MaskingResult:
        """
        Apply typed masks to detected entities.
        
        Args:
            text: Original text.
            entities: Detected entities (from GLiNERDetector).
            post_id: Post identifier for logging.
            
        Returns:
            MaskingResult with masked text and pollution log entries.
        """
        # Sort entities by start position (reverse order for stable masking)
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
        
        masked_text = text
        mask_log = []
        
        for entity in sorted_entities:
            start = entity["start"]
            end = entity["end"]
            entity_type = entity["label"]
            confidence = entity["score"]
            span_text = entity["text"]
            
            # Get typed mask token
            mask_token = self.entity_to_mask.get(entity_type, "[MASK:UNKNOWN]")
            
            # Apply mask (reverse order ensures character offsets remain valid)
            masked_text = masked_text[:start] + mask_token + masked_text[end:]
            
            # Create log entry for POLLUTION_LOG_SCHEMA
            log_entry = {
                "post_id": post_id,
                "span_start": start,
                "span_end": end,
                "span_text": span_text,
                "entity_type": entity_type,
                "confidence": confidence,
                "mask_token": mask_token,
            }
            mask_log.append(log_entry)
        
        logger.debug(f"Masked {len(sorted_entities)} spans in post {post_id}")
        
        return MaskingResult(
            masked_text=masked_text,
            mask_log=mask_log,
        )
    
    def mask_batch(
        self,
        texts: List[str],
        entities_batch: List[List[Dict[str, Any]]],
        post_ids: List[str],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Apply typed masks to a batch of texts.
        
        Args:
            texts: List of original texts.
            entities_batch: List of detected entities per text.
            post_ids: List of post identifiers.
            
        Returns:
            Tuple of (masked_texts, all_mask_logs).
        """
        masked_texts = []
        all_mask_logs = []
        
        for text, entities, post_id in zip(texts, entities_batch, post_ids):
            result = self.mask_spans(text, entities, post_id)
            masked_texts.append(result.masked_text)
            all_mask_logs.extend(result.mask_log)
        
        return masked_texts, all_mask_logs
