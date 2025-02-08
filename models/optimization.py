#!/usr/bin/env python
"""
Gradient Centralization wrapper and GCAdam optimizer.

This implementation centralizes gradients (subtracting the mean along all dimensions except the first)
before performing an optimizer step.
"""

import torch
import torch.optim as optim

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

def GCAdam(params, lr, weight_decay):
    """
    Instantiate a standard Adam optimizer and wrap it with gradient centralization.
    """
    # Instantiate a standard Adam optimizer using the provided learning rate and weight decay
    adam_optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    # Wrap the Adam optimizer with our GradientCentralization class.
    # This wrapper subtracts the mean gradient from parameters with multidimensional gradients (all except batch)
    # in accordance with the guidelines in the notepad.
    return GradientCentralization(adam_optimizer) 