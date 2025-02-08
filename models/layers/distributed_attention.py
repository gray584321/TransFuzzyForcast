#!/usr/bin/env python
"""
DistributedSparseAttention

This module implements a multi-head attention layer using an asymmetric exponential kernel
and a ProbSparse self-attention mechanism. In a full implementation, one would integrate
ProbSparse query selection for efficiency. This version computes a sparsity measure for each
query, selects the top-u queries to compute accurate attention, and fills the remaining queries
with a global average as a default.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributedSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_factor, dropout):
        """
        Distributed Sparse Attention layer.
        This is a simplified placeholder implementation of the ProbSparse attention mechanism.
        Args:
            d_model (int): Hidden dimension.
            n_heads (int): Number of attention heads.
            attention_factor (float): Factor controlling sparse attention selection (used to determine u = c * log(seq_len)).
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_factor = attention_factor
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Linear projections for queries, keys, and values
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, queries, keys, values):
        """
        Compute distributed sparse attention.
        This implementation:
          1. Projects inputs into Q, K, V and splits them into multiple heads.
          2. Computes full attention scores for each head.
          3. For each query computes importance = max(score) - mean(score) (over keys).
          4. Selects top-u queries per head (u = max(1, int(attention_factor * log(seq_len))) capped by seq_len).
          5. Computes accurate attention only for selected queries using an asymmetric exponential kernel.
          6. For non-selected queries, assigns a default value (the global average of V).
          7. Scatters the computed results into the full output tensor.
          8. Merges the heads and applies a final linear projection.
          
        Args:
            queries, keys, values: Tensors of shape (batch, seq_len, d_model)
        Returns:
            Tensor: Attention output of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = queries.size()
        # Linear projections
        Q = self.q_linear(queries)  # (B, seq_len, d_model)
        K = self.k_linear(keys)
        V = self.v_linear(values)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute full scaled dot-product scores: (B, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Compute the sparsity measure for each query:
        # measure = max(score_row) - mean(score_row) for each query along the keys dimension
        importance = scores.max(dim=-1)[0] - scores.mean(dim=-1)  # (B, n_heads, seq_len)
        
        # Determine the number of queries to select: u = max(1, int(attention_factor * log(seq_len)))
        u = max(1, int(self.attention_factor * math.log(seq_len)))
        u = min(u, seq_len)  # Ensure u does not exceed seq_len
        
        # For each head, select the indices of the top-u queries based on the importance measure
        selected_values, selected_indices = importance.topk(k=u, dim=-1)  # (B, n_heads, u)
        
        # Gather the selected queries
        # selected_indices: (B, n_heads, u) -> expand to (B, n_heads, u, head_dim)
        Q_selected = torch.gather(Q, dim=2, index=selected_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        
        # Compute scores for selected queries: (B, n_heads, u, seq_len)
        scores_selected = torch.matmul(Q_selected, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Subtract maximum value for numerical stability before taking exp
        scores_selected_stable = scores_selected - scores_selected.max(dim=-1, keepdim=True)[0]
        # Apply the asymmetric exponential kernel in a stable manner
        kernel = torch.exp(scores_selected_stable)
        weights_selected = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
        weights_selected = self.dropout(weights_selected)
        
        # Compute attention output for selected queries: (B, n_heads, u, head_dim)
        output_selected = torch.matmul(weights_selected, V)
        
        # For non-selected queries, compute a default value (global average of V across the seq_len dimension)
        default_value = V.mean(dim=2, keepdim=True)  # (B, n_heads, 1, head_dim)
        # Initialize output tensor with default values replicated along the query dimension
        out = default_value.expand(batch_size, self.n_heads, seq_len, self.head_dim).clone()  # (B, n_heads, seq_len, head_dim)
        
        # Scatter the computed output for selected queries back into the full tensor.
        # selected_indices is (B, n_heads, u) and needs to be expanded to match output_selected.
        index_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        out.scatter_(dim=2, index=index_expanded, src=output_selected)
        
        # Merge heads: transpose back and reshape to (B, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.out_linear(out)
        return output 