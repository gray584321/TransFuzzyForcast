#!/usr/bin/env python
"""
Gradient Centralization wrapper and GCAdam optimizer.

This implementation centralizes gradients (subtracting the mean along all dimensions except the first)
before performing an optimizer step.
"""

import torch
from torch.optim import Adam

class GradientCentralization:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # Apply gradient centralization before taking an optimizer step
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None and param.grad.dim() > 1:
                    # Compute mean of gradients along dimensions except the first
                    grad_mean = param.grad.mean(dim=tuple(range(1, param.grad.dim())), keepdim=True)
                    param.grad.data.sub_(grad_mean)
        self.optimizer.step()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

# Updated GCAdam: Now inherits from torch.optim.Adam so that it is recognized
# as an optimizer by torch.optim.lr_scheduler and other components.
class GCAdam(Adam):  # <-- Changed inheritance here.
    def __init__(self, params, lr=1e-3, weight_decay=0, **kwargs):
        super().__init__(params, lr=lr, weight_decay=weight_decay, **kwargs)

    def step(self, closure=None):
        # Before taking the optimization step, apply gradient centralization.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Apply gradient centralization if the gradient has more than 1 dimension.
                if p.grad.dim() > 1:
                    # Subtract the mean of gradients along all dimensions except the first one.
                    p.grad.data = p.grad.data - p.grad.data.mean(dim=tuple(range(1, p.grad.data.dim())), keepdim=True)
        return super().step(closure) 