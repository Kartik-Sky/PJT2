import torch
import numpy as np
import torch.nn as nn
from config.ModelSettings import CMSConfig


class CMSBlock(nn.Module):
    """A single Conditional Memory System block with an MLP residual connection.
 
    Each block has an associated update frequency that determines how often
    its parameters should be updated during training.
 
    Args:
        config: A CMSConfig instance providing shared architectural settings.
        update_frequency: How often (in steps) this block's parameters should be updated.
        hidden_multiplier: Multiplier for the hidden layer size in the MLP.
    """
 
    def __init__(self, config: CMSConfig, update_frequency: int, hidden_multiplier: int):
        super().__init__()
        self.dim = config.dim
        self.hidden_multiplier = hidden_multiplier
 
        layers = [
            nn.Linear(config.dim, config.dim * hidden_multiplier),
            config.activation(),
            nn.Linear(config.dim * hidden_multiplier, config.dim),
        ]
        if config.use_batch_norm:
            layers.append(nn.LayerNorm(config.dim))
 
        self.mlp = nn.Sequential(*layers)
        self._update_frequency = update_frequency
 
    def should_update(self, idx: int) -> bool:
        """Check whether this block should be updated at the given step index.
 
        Args:
            idx: The current global training step.
 
        Returns:
            True if the block should be updated at this step, False otherwise.
        """
        return idx % self._update_frequency == 0
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with a residual MLP connection.
 
        Args:
            x: Input tensor of shape (batch_size, dim).
 
        Returns:
            Output tensor of shape (batch_size, dim) with residual added.
        """
        B, S, H = x.shape

        x_reshaped = x.view(B * S, H)     # (B*S, H)
        out = self.mlp(x_reshaped)
        out = out.view(B, S, H)

        return x + out
 
 
class CMSNet(nn.Module):
    """A network composed of multiple CMSBlocks with heterogeneous update frequencies.
 
    Each block can have a different update frequency and hidden multiplier,
    enabling multi-scale parameter updates during training.
 
    Args:
        config: A CMSConfig instance fully describing the network architecture.
    """
 
    def __init__(self, config: CMSConfig):
        super().__init__()
        self.config = config
 
        self.blocks = nn.ModuleDict(
            {
                f"{config.block_name_prefix}_{i + 1}": CMSBlock(
                    config,
                    config.update_frequencies[i],
                    config.multipliers[i],
                )
                for i in range(config.num_blocks)
            }
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks with a detached residual connection.
 
        The input is detached before passing through blocks to prevent
        gradient flow through the residual path.
 
        Args:
            x: Input tensor of shape (batch_size, dim).
 
        Returns:
            Output tensor of shape (batch_size, dim).
        """
        z = x.detach()
        for block in self.blocks.values():
            z = block(z)
        return x + z
 
    def get_param_groups(self) -> dict:
        """Return all learnable parameter groups with their update frequencies.
 
        Returns:
            Dictionary mapping block names to tuples of (update_frequency, param_list).
        """
        return {
            name: (block._update_frequency, [p for p in block.parameters() if p.requires_grad])
            for name, block in self.blocks.items()
        }
 
    def get_update_param_groups(self, idx: int) -> dict:
        """Return only the parameter groups that should be updated at the given step.
 
        Args:
            idx: The current global training step.
 
        Returns:
            Dictionary mapping block names to their learnable parameters,
            filtered to only include blocks due for update at this step.
        """
        return {
            name: [p for p in block.parameters() if p.requires_grad]
            for name, block in self.blocks.items()
            if block.should_update(idx)
        }
 
    def should_update(self, key: str):
        """Return the should_update method of the specified block.
 
        Args:
            key: The block name to look up.
 
        Returns:
            The block's should_update method, or False if the key is not found.
        """
        if key in self.blocks:
            return self.blocks[key].should_update
        return False
 