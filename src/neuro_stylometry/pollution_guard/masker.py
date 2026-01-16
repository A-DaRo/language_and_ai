"""
Span Masker for Pollution Mitigation.

Applies typed masks to detected pollution spans in a token-aware manner.

Implements: FR-08 (Safe Masking), GLiNER_Implementation_Strategy.md Section 2.3
"""

import logging
from pathlib import Path
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
    - Config-driven: mask tokens extracted from gliner_taxonomy.yaml
    - Maps entity types (column names) to typed mask tokens
    - Applies masks in reverse order to preserve character offsets
    - Validates single-token encoding of masks
    - Emits structured mask logs for POLLUTION_LOG_SCHEMA
    
    Implements: FR-08
    
    Preferred construction:
        >>> masker = SpanMasker.from_taxonomy(taxonomy_cfg, tokenizer=tokenizer)
    """
    
    def __init__(
        self,
        tokenizer=None,
        entity_to_mask: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize masker.
        
        Args:
            tokenizer: Optional HuggingFace tokenizer for validation.
            entity_to_mask: Entity/column -> mask mapping.
                When using from_taxonomy(), this is populated from YAML.
        """
        self.tokenizer = tokenizer
        self.entity_to_mask = entity_to_mask if entity_to_mask else {}
        
        # Validate single-token encoding if tokenizer provided
        if self.tokenizer and self.entity_to_mask:
            self._validate_mask_tokens()
    
    @classmethod
    def from_taxonomy(
        cls,
        taxonomy_cfg: Dict[str, Any],
        tokenizer=None,
    ) -> "SpanMasker":
        """
        Create SpanMasker from taxonomy configuration (preferred factory).
        
        Extracts mask_token from each column definition in gliner_taxonomy.yaml.
        Also maps GLiNER prompt labels to their corresponding mask tokens.
        
        Args:
            taxonomy_cfg: Loaded taxonomy dict with 'column_prompts' section.
                Expected structure:
                    column_prompts:
                      birth_year:
                        prompts: ["age statement", ...]
                        mask_token: "[MASK:AGE]"
                      female:
                        prompts: ["gender self-identification", ...]
                        mask_token: "[MASK:GENDER]"
                      ...
            tokenizer: Optional HuggingFace tokenizer for single-token validation.
            
        Returns:
            Configured SpanMasker instance.
            
        Raises:
            ValueError: If taxonomy_cfg is missing required structure.
        """
        entity_to_mask: Dict[str, str] = {}
        
        column_prompts = taxonomy_cfg.get("column_prompts", {})
        if not column_prompts:
            logger.warning(
                "No 'column_prompts' found in taxonomy_cfg; "
                "SpanMasker will have empty entity_to_mask mapping"
            )
        
        for column_name, column_config in column_prompts.items():
            if not isinstance(column_config, dict):
                logger.warning(f"Invalid config for column '{column_name}': expected dict")
                continue
            
            mask_token = column_config.get("mask_token")
            if not mask_token:
                logger.warning(f"No mask_token defined for column '{column_name}'")
                continue
            
            # Map column name to mask token
            entity_to_mask[column_name] = mask_token
            
            # Also map each prompt label to the same mask token
            # This handles GLiNER entity types like "age statement" -> "[MASK:AGE]"
            prompts = column_config.get("prompts", [])
            for prompt in prompts:
                if isinstance(prompt, str):
                    # Normalize prompt to entity label format (underscore-separated)
                    entity_label = prompt.lower().replace(" ", "_")
                    entity_to_mask[entity_label] = mask_token
                    # Also keep the original prompt format
                    entity_to_mask[prompt] = mask_token
            
            # Map distractors too (they might be detected)
            distractors = column_config.get("distractors", [])
            for distractor in distractors:
                if isinstance(distractor, str):
                    entity_label = distractor.lower().replace(" ", "_")
                    entity_to_mask[entity_label] = mask_token
                    entity_to_mask[distractor] = mask_token
        
        logger.info(
            f"SpanMasker.from_taxonomy: loaded {len(entity_to_mask)} "
            f"entity->mask mappings from {len(column_prompts)} columns"
        )
        
        return cls(tokenizer=tokenizer, entity_to_mask=entity_to_mask)
    
    @classmethod
    def from_taxonomy_path(
        cls,
        taxonomy_path: Path,
        tokenizer=None,
    ) -> "SpanMasker":
        """
        Create SpanMasker from taxonomy YAML file path.
        
        Convenience wrapper around from_taxonomy() that loads the YAML file.
        
        Args:
            taxonomy_path: Path to gliner_taxonomy.yaml file.
            tokenizer: Optional HuggingFace tokenizer for validation.
            
        Returns:
            Configured SpanMasker instance.
        """
        from omegaconf import OmegaConf
        
        taxonomy_path = Path(taxonomy_path)
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy config not found: {taxonomy_path}")
        
        cfg = OmegaConf.to_container(OmegaConf.load(taxonomy_path), resolve=True)
        if not isinstance(cfg, dict):
            raise TypeError("Taxonomy YAML did not resolve to a mapping")
        
        taxonomy_cfg = cfg.get("taxonomy", cfg)
        return cls.from_taxonomy(taxonomy_cfg, tokenizer=tokenizer)
    
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
