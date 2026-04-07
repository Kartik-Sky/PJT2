from dataclasses import dataclass, field
from typing import List, Callable
import torch.nn as nn


@dataclass
class CMSConfig:
    """Configuration dataclass for initializing a CMSNet.
 
    Args:
        dim: Feature dimension shared across all blocks.
        num_blocks: Number of CMSBlocks in the network.
        update_frequencies: List of update frequencies, one per block.
            Each value controls how often (in steps) that block's parameters
            are updated during training.
        multipliers: List of hidden layer multipliers, one per block.
            Controls the intermediate MLP width as `dim * multiplier`.
        activation: Activation function used inside each block's MLP.
            Defaults to nn.GELU.
        use_batch_norm: Whether to apply BatchNorm1d after the MLP projection.
            Defaults to True.
        block_name_prefix: Prefix used when naming blocks in the ModuleDict.
            Defaults to "block".
    """
 
    dim: int
    num_blocks: int
    update_frequencies: List[int]
    multipliers: List[int]
    activation: Callable[[], nn.Module] = field(default_factory=lambda: nn.GELU)
    use_batch_norm: bool = True
    block_name_prefix: str = "block"
 
    def __post_init__(self):
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if len(self.update_frequencies) != self.num_blocks:
            raise ValueError("update_frequencies length must match num_blocks")
        if len(self.multipliers) != self.num_blocks:
            raise ValueError("multipliers length must match num_blocks")
        if any(f < 1 for f in self.update_frequencies):
            raise ValueError("All update_frequencies must be >= 1")
        if any(m < 1 for m in self.multipliers):
            raise ValueError("All multipliers must be >= 1")
 
