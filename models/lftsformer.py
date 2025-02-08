#!/usr/bin/env python
"""
EnhancedLFTSformer Model

This file implements the DS Encoder Informer architecture:
  - InputEmbedding: Projects input features, adds sinusoidal positional encoding (and optional timestamp embedding),
    and applies a Conv1D transformation.
  - DS_EncoderLayer: A Transformer encoder layer with distributed sparse attention and a feed-forward network.
  - StackedInformerEncoder: Processes the input embedding over three scales (L, L/2, L/4) with downsampling and fuses features.
  - EnhancedLFTSformer: Combines the embedding and encoder, then produces final predictions via an output layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the distributed sparse attention layer from models/layers/distributed_attention.py
from .layers.distributed_attention import DistributedSparseAttention

class InputEmbedding(nn.Module):
    def __init__(self, feature_dim, d_model, max_seq_len, timestamp_vocab_size=None, debug=False):
        """
        Args:
            feature_dim (int): Number of input features.
            d_model (int): Model hidden dimensionality.
            max_seq_len (int): Maximum sequence length.
            timestamp_vocab_size (int, optional): Size of the timestamp vocabulary. If provided, a global timestamp embedding is added.
            debug (bool): If True, prints intermediate debug info.
        """
        super().__init__()
        self.debug = debug
        # Scalar projection using a single linear layer
        self.scalar_projection = nn.Linear(feature_dim, d_model)
        # Positional encoding (non-learnable), registered as buffer
        self.register_buffer("positional_encoding", self._get_positional_encoding(max_seq_len, d_model))
        self.timestamp_vocab_size = timestamp_vocab_size
        if timestamp_vocab_size is not None:
            self.timestamp_embedding = nn.Embedding(timestamp_vocab_size, d_model)
        else:
            self.timestamp_embedding = None
        # Feature transformation: Conv1D layer with kernel_size=1 acts as a linear transformation across the time dimension
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def _get_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        return pe

    def forward(self, x, timestamps=None):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, feature_dim).
            timestamps (Tensor, optional): Timestamp indices of shape (batch, seq_len).
        Returns:
            Tensor: Embedded tensor of shape (batch, seq_len, d_model).
        """
        # Validate input feature dimension
        if x.size(-1) != self.scalar_projection.in_features:
            raise ValueError(f"Input feature dimension mismatch: expected {self.scalar_projection.in_features}, but got {x.size(-1)}")

        if self.debug:
            print(f"[InputEmbedding] Input x: shape {x.shape}, min {x.min().item():.4f}, max {x.max().item():.4f}, nan {torch.isnan(x).any().item()}")

        batch_size, seq_len, _ = x.size()
        projected = self.scalar_projection(x)  # (batch, seq_len, d_model)
        if self.debug:
            print(f"[InputEmbedding] After scalar projection: shape {projected.shape}, mean {projected.mean().item():.4f}, std {projected.std().item():.4f}, nan {torch.isnan(projected).any().item()}")

        transformed = self.conv1(projected.transpose(1, 2)).transpose(1, 2)  # (batch, seq_len, d_model)
        if self.debug:
            print(f"[InputEmbedding] After conv1 transformation: shape {transformed.shape}, mean {transformed.mean().item():.4f}, std {transformed.std().item():.4f}, nan {torch.isnan(transformed).any().item()}")

        pe = self.positional_encoding[:, :seq_len, :]
        out = transformed + pe
        if self.debug:
            print(f"[InputEmbedding] After adding positional encoding: shape {out.shape}, mean {out.mean().item():.4f}, std {out.std().item():.4f}, nan {torch.isnan(out).any().item()}")

        if self.timestamp_embedding is not None and timestamps is not None:
            tse = self.timestamp_embedding(timestamps)  # (batch, seq_len, d_model)
            out = out + tse
            if self.debug:
                print(f"[InputEmbedding] After adding timestamp embedding: shape {out.shape}, mean {out.mean().item():.4f}, std {out.std().item():.4f}, nan {torch.isnan(out).any().item()}")

        return out

class DS_EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, attention_factor, dropout):
        """
        A single encoder layer with distributed sparse attention and a feed-forward network.
        """
        super().__init__()
        self.attention = DistributedSparseAttention(d_model, n_heads, attention_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).
        Returns:
            Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        attn_output = self.attention(x, x, x)  # Self-attention
        x = x + attn_output
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output
        x = self.norm2(x)
        return x

class StackedInformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, attention_factor, dropout,
                 num_layers_l=3, num_layers_lhalf=2, num_layers_lquarter=1):
        """
        Stacked Informer Encoder that processes input features at three temporal scales.
        """
        super().__init__()
        # L-scale layers
        self.l_scale_layers = nn.ModuleList([
            DS_EncoderLayer(d_model, d_ff, n_heads, attention_factor, dropout)
            for _ in range(num_layers_l)
        ])
        # Downsampling from L scale to L/2 using Conv1D with stride=2
        self.conv_down1 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        
        # L/2-scale layers
        self.l_half_scale_layers = nn.ModuleList([
            DS_EncoderLayer(d_model, d_ff, n_heads, attention_factor, dropout)
            for _ in range(num_layers_lhalf)
        ])
        # Downsampling from L/2 to L/4
        self.conv_down2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        
        # L/4-scale layers
        self.l_quarter_scale_layers = nn.ModuleList([
            DS_EncoderLayer(d_model, d_ff, n_heads, attention_factor, dropout)
            for _ in range(num_layers_lquarter)
        ])
        # Fusion layer to combine multi-scale outputs
        self.fusion_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Embedded input from InputEmbedding (batch, seq_len, d_model)
        Returns:
            Tensor: Unified encoder representation of shape (batch, target_len, d_model)
        """
        # L-scale processing
        l_scale_out = x
        for layer in self.l_scale_layers:
            l_scale_out = layer(l_scale_out)

        # Downsample L-scale to L/2
        l_scale_down = self.conv_down1(l_scale_out.transpose(1, 2)).transpose(1, 2)
        
        # L/2-scale processing
        l_half_scale_out = l_scale_down
        for layer in self.l_half_scale_layers:
            l_half_scale_out = layer(l_half_scale_out)
        
        # Downsample L/2-scale to L/4
        l_half_down = self.conv_down2(l_half_scale_out.transpose(1, 2)).transpose(1, 2)
        
        # L/4-scale processing
        l_quarter_scale_out = l_half_down
        for layer in self.l_quarter_scale_layers:
            l_quarter_scale_out = layer(l_quarter_scale_out)
        
        # Fuse outputs from different scales by first aggregating to a common sequence length
        target_len = l_quarter_scale_out.size(1)
        l_scale_fused = F.adaptive_avg_pool1d(l_scale_out.transpose(1, 2), output_size=target_len).transpose(1, 2)
        l_half_fused = F.adaptive_avg_pool1d(l_half_scale_out.transpose(1, 2), output_size=target_len).transpose(1, 2)
        # Fusion by summation and projection
        fused = l_scale_fused + l_half_fused + l_quarter_scale_out
        fused = self.fusion_linear(fused)
        return fused

class EnhancedLFTSformer(nn.Module):
    def __init__(self, feature_dim, d_model, d_ff, n_heads, attention_factor, dropout,
                 max_seq_len, pred_len, output_dim, timestamp_vocab_size=None, debug=False):
        """
        The full DS Encoder Informer model.
        
        Args:
            feature_dim (int): Input feature dimension.
            d_model (int): Hidden dimension.
            d_ff (int): FFN hidden dimension.
            n_heads (int): Number of attention heads.
            attention_factor (float): Factor controlling sparse attention selection.
            dropout (float): Dropout rate.
            max_seq_len (int): Maximum input sequence length.
            pred_len (int): Number of time steps to predict.
            output_dim (int): Number of output features (e.g., 1 for closing price).
            timestamp_vocab_size (int, optional): Vocabulary size for global timestamp embedding.
            debug (bool): If True, prints intermediate debug info.
        """
        super().__init__()
        self.debug = debug
        self.input_embedding = InputEmbedding(feature_dim, d_model, max_seq_len, timestamp_vocab_size, debug=debug)
        self.encoder = StackedInformerEncoder(d_model, d_ff, n_heads, attention_factor, dropout)
        self.output_layer = nn.Linear(d_model, pred_len * output_dim)
        self.pred_len = pred_len
        self.output_dim = output_dim

    def forward(self, x, timestamps=None):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, feature_dim).
            timestamps (Tensor, optional): Tensor of timestamp indices (batch, seq_len).
        Returns:
            Tensor: Predictions shaped as (batch, pred_len, output_dim).
        """
        # Ensure input is 3D: [batch_size, seq_len, feature_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if timestamps is not None:
                timestamps = timestamps.unsqueeze(1)
        
        if self.debug:
            print(f"[EnhancedLFTSformer] Input x: shape {x.shape}, nan {torch.isnan(x).any().item()}")

        embed = self.input_embedding(x, timestamps)
        if self.debug:
            print(f"[EnhancedLFTSformer] After input embedding: shape {embed.shape}, mean {embed.mean().item():.4f}, std {embed.std().item():.4f}, nan {torch.isnan(embed).any().item()}")

        encoder_out = self.encoder(embed)
        if self.debug:
            print(f"[EnhancedLFTSformer] After encoder: shape {encoder_out.shape}, mean {encoder_out.mean().item():.4f}, std {encoder_out.std().item():.4f}, nan {torch.isnan(encoder_out).any().item()}")

        pooled = encoder_out.mean(dim=1)
        if self.debug:
            print(f"[EnhancedLFTSformer] After pooling: shape {pooled.shape}, mean {pooled.mean().item():.4f}, std {pooled.std().item():.4f}, nan {torch.isnan(pooled).any().item()}")

        output = self.output_layer(pooled)
        if self.debug:
            print(f"[EnhancedLFTSformer] After output layer before reshaping: shape {output.shape}, mean {output.mean().item():.4f}, std {output.std().item():.4f}, nan {torch.isnan(output).any().item()}")

        output = output.view(output.size(0), self.pred_len, self.output_dim)
        if self.debug:
            print(f"[EnhancedLFTSformer] Final output: shape {output.shape}, mean {output.mean().item():.4f}, std {output.std().item():.4f}, nan {torch.isnan(output).any().item()}")
        return output 