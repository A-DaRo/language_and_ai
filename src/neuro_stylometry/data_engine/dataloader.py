"""
DataLoaderFactory: Create PyTorch DataLoaders with hardware-aware configuration.

Handles multi-task collation, tokenization, and NULL value mapping for demographic labels.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
from datasets import Dataset
from transformers import PreTrainedTokenizer
import logging

from .schemas import get_demographic_columns, get_nullable_columns

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory for creating PyTorch DataLoaders with appropriate configuration.
    
    Responsibilities:
        - Multi-task collation (convert demographic labels to tensors)
        - NULL value handling (map to ignore_index=-100)
        - Hardware-aware worker configuration
        - Tokenization integration
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        hardware_profile: str = "laptop",
    ):
        """
        Initialize factory with tokenizer and hardware profile.
        
        Args:
            tokenizer: HuggingFace tokenizer (e.g., RobertaTokenizer).
            max_length: Maximum sequence length for tokenization.
            hardware_profile: One of "laptop", "hpc", "generic".
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hardware_profile = hardware_profile
        
        # Hardware-specific configurations
        self.worker_configs = {
            'laptop': {
                'num_workers': 2,
                'pin_memory': False,
                'prefetch_factor': 2,
                'persistent_workers': False,
            },
            'hpc': {
                'num_workers': 8,
                'pin_memory': True,
                'prefetch_factor': 4,
                'persistent_workers': True,
            },
            'generic': {
                'num_workers': 4,
                'pin_memory': True,
                'prefetch_factor': 2,
                'persistent_workers': False,
            }
        }
    
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        sampler: Optional[torch.utils.data.Sampler] = None,
    ) -> DataLoader:
        """
        Create a DataLoader for the given dataset.
        
        Args:
            dataset: HuggingFace Dataset instance.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data (ignored if sampler is provided).
            sampler: Optional custom sampler (e.g., BucketSampler).
            
        Returns:
            Configured PyTorch DataLoader.
        """
        # Get hardware-specific configuration
        config = self.worker_configs.get(
            self.hardware_profile,
            self.worker_configs['generic']
        )
        
        # Create collate function
        collate_fn = self._create_collate_fn()
        
        # Build DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collate_fn,
            **config
        )
        
        logger.info(f"Created DataLoader: batch_size={batch_size}, "
                   f"workers={config['num_workers']}, "
                   f"profile={self.hardware_profile}")
        
        return dataloader
    
    def _create_collate_fn(self) -> Callable:
        """
        Create a collate function for batching.
        
        Returns:
            Collate function that handles tokenization and multi-task labels.
        """
        def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            """
            Collate a batch of samples.
            
            Args:
                batch: List of dictionaries from HuggingFace Dataset.
                
            Returns:
                Dictionary with tokenized inputs and label tensors.
            """
            # Extract text for tokenization
            texts = [sample['post'] for sample in batch]
            
            # Tokenize
            encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Initialize output dictionary
            output = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'post_id': [sample['post_id'] for sample in batch],
            }
            
            # Add demographic labels (handle NULLs)
            demographic_cols = get_demographic_columns()
            for col in demographic_cols:
                labels = []
                for sample in batch:
                    value = sample.get(col)
                    
                    # Handle NULL values
                    if value is None or (isinstance(value, float) and torch.isnan(torch.tensor(value))):
                        labels.append(-100)  # PyTorch ignore_index
                    else:
                        # Convert categorical to integer index
                        if isinstance(value, str):
                            # For categorical columns, we need to map to integer
                            # This assumes the dataset has already converted categories to indices
                            # If not, we'd need a label encoder here
                            labels.append(int(hash(value) % 1000))  # Placeholder mapping
                        else:
                            labels.append(int(value))
                
                output[f'{col}_labels'] = torch.tensor(labels, dtype=torch.long)
            
            return output
        
        return collate_batch


def create_tokenizing_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 512,
    hardware_profile: str = "laptop",
    shuffle: bool = True,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> DataLoader:
    """
    Convenience function to create a tokenizing DataLoader.
    
    Args:
        dataset: HuggingFace Dataset.
        tokenizer: PreTrainedTokenizer instance.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        hardware_profile: Hardware execution profile.
        shuffle: Whether to shuffle.
        sampler: Optional custom sampler.
        
    Returns:
        Configured DataLoader.
    """
    factory = DataLoaderFactory(
        tokenizer=tokenizer,
        max_length=max_length,
        hardware_profile=hardware_profile
    )
    
    return factory.create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler
    )
