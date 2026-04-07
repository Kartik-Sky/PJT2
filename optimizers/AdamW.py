import torch
from torch.optim import Optimizer
import numpy as np


class CMSAdamW(Optimizer):
    """AdamW optimizer adapted for Conditional Memory System (CMS) parameter groups.
    
    Supports per-group update frequencies, allowing different parameter groups
    to be updated at different intervals. Implements decoupled weight decay
    regularization with bias-corrected first and second moment estimates.
    
    Args:
        params: Iterable of parameter group dicts with 'params' and 'frequency' keys.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps: Term added to denominator for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.global_step = 0
        super().__init__(params, defaults)

    def _init_state(self, p):
        """Initialize optimizer state for a parameter if not already initialized.
        
        Args:
            p: The parameter tensor.
            
        Returns:
            The optimizer state dictionary for this parameter, containing
            'step', 'exp_avg', and 'exp_avg_sq'.
        """
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step using AdamW update rule.
        
        Only updates parameter groups whose frequency aligns with the current global step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).
            
        Returns:
            The loss value if a closure was provided, otherwise None.
        """

        self.global_step += 1


        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            if ('frequency' not in group.keys()):
                continue
            freq = group['frequency']
            if (self.global_step % freq != 0):
                continue
            
            # LOGIC starts here
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self._init_state(p)
                state['step'] += 1

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Update biased first and second moment estimates
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                corrected_avg = exp_avg / bias_correction1
                corrected_avg_sq = exp_avg_sq / bias_correction2

                # Parameter update
                p.add_(corrected_avg / (corrected_avg_sq.sqrt() + eps), alpha=-lr)
                
        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none=True):
        """Zero out gradients only for parameter groups not scheduled for update.
        
        Preserves gradients for groups that are due for an update at the current
        global step, allowing gradient accumulation across steps.
        
        Args:
            set_to_none: If True, set gradients to None instead of zeroing (default: True).
        """
        for group in self.param_groups:
            
            if ('frequency' not in group.keys()):
                for p in group['params']:
                    
                    if set_to_none:
                        p.grad=None
                    else:
                        p.grad.zero_()

            else:
                group_freq = group['frequency']
                if (self.global_step % group_freq == 0):
                    continue
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad=None
                        else:
                            p.grad.zero_()


# class CMSAdamW(Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), 
#                  eps=1e-8, weight_decay=0.0, accum_steps=1):
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         self.global_step = 0
#         self.accum_steps = accum_steps  # store it
#         super().__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, closure=None):
#         self.global_step += 1

#         # Effective step accounts for accumulation
#         effective_step = self.global_step * self.accum_steps

#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             if 'frequency' not in group:
#                 continue
#             freq = group['frequency']
#             if effective_step % freq != 0:
#                 continue
#             # ... rest of update unchanged

#     @torch.no_grad()
#     def zero_grad(self, set_to_none=True):
#         effective_step = self.global_step * self.accum_steps

#         for group in self.param_groups:
#             if 'frequency' not in group:
#                 for p in group['params']:
#                     p.grad = None if set_to_none else p.grad.zero_()
#             else:
#                 group_freq = group['frequency']
#                 if effective_step % group_freq == 0:
#                     continue  # preserve gradients — update due
#                 for p in group['params']:
#                     if p.grad is not None:
#                         p.grad = None if set_to_none else p.grad.zero_()


