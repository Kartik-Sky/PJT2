import math
import torch
import numpy as np
from torch.optim import Optimizer

class M3(Optimizer):
    """Multi-Moment Muon (M3) optimizer with orthogonalized momentum directions.
    
    Combines a fast first-moment estimate (M1) with a slower accumulated
    gradient signal (M2), both orthogonalized before computing the update.
    Uses an adaptive second-moment denominator similar to Adam.
    
    Args:
        params: Iterable of parameter group dicts with 'params' and 'frequency' keys.
        f: Frequency (in steps) for updating the slow momentum M2. Required.
        lr: Learning rate (default: 1e-3).
        betas: Tuple of (beta1, beta2, beta3) for M1, V, and M2 updates (default: (0.9, 0.95, 0.999)).
        alpha: Blending coefficient for the slow orthogonal direction O2 (default: 0.5).
        eps: Term added to denominator for numerical stability (default: 1e-8).
        weight_decay: Weight decay coefficient (default: 0.0).
        ortho_method: Orthogonalization method, either 'nschulz' or 'svd' (default: 'nschulz').
    """

    def __init__(self, params, f, lr=1e-3, betas=(0.9, 0.95, 0.999), alpha=0.5, eps=1e-8, weight_decay=0.0, ortho_method='nschulz'):
        if f is None:
            raise ValueError("f is required and cannot be None")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta[2]: {betas[2]}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ortho_method == 'svd':
            self._orthogonalize = self._orthogonalize_svd
        elif ortho_method == 'nschulz':
            self._orthogonalize = self._orthogonalize_newton_schulz

        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, f=f, weight_decay=weight_decay)
        self.global_step = 0

        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize_svd(M):
        """Orthogonalize a matrix using SVD decomposition.
        
        Computes the closest orthogonal matrix via U @ V^T from the
        singular value decomposition of M.
        
        Args:
            M: Input matrix tensor.
            
        Returns:
            Orthogonalized matrix with the same shape as M.
        """
        U, _, Vt = torch.linalg.svd(M, full_matrices=False)
        return U @ Vt

    @staticmethod
    def _orthogonalize_newton_schulz(M, num_iters=5):
        """Orthogonalize a matrix using Newton-Schulz iteration.
        
        Uses the cubic polynomial variant with coefficients (a, b, c) to
        iteratively converge to the closest orthogonal matrix. Handles
        1D tensors by temporarily unsqueezing and reshaping back.
        
        Args:
            M: Input matrix tensor (1D or 2D).
            num_iters: Number of Newton-Schulz iterations (default: 5).
            
        Returns:
            Orthogonalized tensor with the same shape as M.
        """
        a, b, c = (3.4445, -4.7750, 2.0315)
        orig_shape = M.shape
        X = M / (M.norm() + 1e-7)
        if X.ndim < 2:
            X = X.unsqueeze(0)

        for _ in range(num_iters):
            A = X @ X.T
            B = a * torch.eye(A.shape[0], device=A.device, dtype=A.dtype) + b * A + c * A @ A
            X = B @ X

        return X.reshape(orig_shape)

    def _init_state(self, p):
        """Initialize optimizer state for a parameter if not already initialized.
        
        State contains:
            - M1: Fast first-moment accumulator.
            - M2: Slow accumulated gradient momentum.
            - Acc_g: Gradient accumulator between M2 updates.
            - V: Second-moment (squared gradient) accumulator.
            - O2: Cached orthogonalized slow direction.
        
        Args:
            p: The parameter tensor.
            
        Returns:
            The optimizer state dictionary for this parameter.
        """
        state = self.state[p]
        # TODO: implement state initialization
        if len(state) == 0:
            state['M1'] = torch.zeros_like(p.data)
            state['M2'] = torch.zeros_like(p.data)
            state['Acc_g'] = torch.zeros_like(p.data)
            state['V'] = torch.zeros_like(p.data)
            state['O2'] = torch.zeros_like(p.data)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single M3 optimization step.
        
        For each active parameter group (based on frequency and global step):
            1. Accumulates gradients into Acc_g.
            2. Updates M2 and recomputes O2 every f steps.
            3. Updates M1 and computes O1 each step.
            4. Updates the second moment V.
            5. Applies the parameter update using the blended orthogonal direction.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).
            
        Returns:
            The loss value if a closure was provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            freq = group['frequency']
            if (self.global_step % freq != 0):
                continue
            
            lr = group['lr']
            beta1, beta2, beta3 = group['betas']
            alpha = group['alpha']
            eps = group['eps']
            f = group['f']
            group_freq = group['frequency']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                state = self._init_state(p)

                M1 = state['M1']
                M2 = state['M2']
                g_acc = state['Acc_g']
                V = state['V']

                # Accumulate gradient
                g_acc.add_(grad)

                if (self.global_step % f == 0):
                    M2.add_(g_acc, alpha=-beta3)
                    state['O2'] = self._orthogonalize(M2)
                    g_acc.zero_()

                O2 = state['O2']

                # First moment update: M1 = M1 + beta1 * grad
                M1.add_(grad, alpha=beta1)
                O1 = self._orthogonalize(M1)

                # Second moment update: V = V + beta2 * grad^2
                V.addcmul_(grad, grad, value=beta2)

                if (self.global_step % group_freq == 0):
                    # Combined direction: O1 + alpha * O2
                    direction = O1.add(O2, alpha=alpha)
                    # Denominator: sqrt(V + eps)
                    denom = V.add(eps).sqrt_()
                    # Parameter update: p = p - lr * direction / denom
                    p.data.addcdiv_(direction, denom, value=-lr)

        self.global_step += 1

        return loss

    # @torch.no_grad()
    # def zero_grad(self, set_to_none=True):
    #     """Zero out gradients."""
        
    #     pass

