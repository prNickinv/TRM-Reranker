"""Reasoning blocks for TRM Reranker."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from .utils import trunc_normal_init_, rms_norm

CosSin = Tuple[torch.Tensor, torch.Tensor]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: [batch_size, seq_len, num_heads, head_dim]
        k: [batch_size, seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight.to(input_dtype) * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.enabled = base > 0
        if not self.enabled:
            return

        # RoPE frequency computation
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        """Returns cached cos and sin embeddings."""
        if not self.enabled:
            return None, None
        return self.cos_cached, self.sin_cached


class CastedLinear(nn.Module):
    """Linear layer with automatic dtype casting."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((out_features, in_features)),
                std=1.0 / (in_features ** 0.5)
            )
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


class CastedEmbedding(nn.Module):
    """Embedding layer with automatic dtype casting."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float = 0.02,
        cast_to: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.cast_to = cast_to

        # Truncated normal initialization
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cast_to is not None:
            return F.embedding(input, self.embedding_weight.to(self.cast_to))
        return F.embedding(input, self.embedding_weight)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = self._find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)
    
    @staticmethod
    def _find_multiple(a, b):
        return (-(a // -b)) * b
    
    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    """Multi-head self-attention with optional Rotary Position Embeddings."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope

        self.qkv_proj = CastedLinear(hidden_size, hidden_size * 3, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, B, seq_len, num_heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.use_rope and cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for scaled_dot_product_attention: [B, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


@dataclass
class ReasoningBlockConfig:
    """Configuration for reasoning block."""
    hidden_size: int
    num_heads: int
    expansion: float
    rms_norm_eps: float = 1e-5
    dropout: float = 0.1
    use_rope: bool = False


class ReasoningBlock(nn.Module):
    """Single reasoning transformer block."""

    def __init__(self, config: ReasoningBlockConfig):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_rope=config.use_rope,
        )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        #self.rmsnorm1 = RMSNorm(dim=config.hidden_size)
        #self.rmsnorm2 = RMSNorm(dim=config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(hidden_states, attention_mask, cos_sin)
        hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
        #hidden_states = self.rmsnorm1(hidden_states + attn_out)

        # MLP with residual
        mlp_out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
        #hidden_states = self.rmsnorm2(hidden_states + mlp_out)

        return hidden_states


class ReasoningModule(nn.Module):
    """Stack of reasoning blocks."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, cos_sin)
        return hidden_states
