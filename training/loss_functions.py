#!/usr/bin/env python
"""
Dynamic Loss Function

Implements a dynamic loss function f(z, β, c) where:
  - z = predicted - target.
  - If |z| < β, a squared error (scaled by β) is used,
  - Else a linear penalty with slope c is applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLossFunction(nn.Module):
    def __init__(self, beta_initial, c):
        """
        Dynamic Loss Function with robustness parameter 'beta' and constant 'c'
        Args:
            beta_initial (float): Initial value for beta.
            c (float): Constant as described in the dynamic loss formulation.
        """
        super(DynamicLossFunction, self).__init__()
        # For simplicity, beta is treated as a fixed scalar here.
        self.beta = beta_initial
        self.c = c

    def forward(self, predicted, target):
        """
        Compute dynamic loss.
        For demonstration, using a variant of MAE combined with a squared error term.
        Replace with the full Equation 37 as needed.
        """
        # Compute the residual
        z = predicted - target
        # Example dynamic loss: |z|^beta + c * z^2
        loss = torch.mean(torch.abs(z) ** self.beta + self.c * z**2)
        return loss

    def get_current_beta(self):
        """Returns the current beta value as a Python float"""
        return self.beta.item() 