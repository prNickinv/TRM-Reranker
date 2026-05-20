"""Utility functions for modules."""

import math
import torch


def trunc_normal_init_(
    tensor: torch.Tensor, 
    std: float = 1.0, 
    lower: float = -2.0, 
    upper: float = 2.0
) -> torch.Tensor:
    """
    Truncated normal initialization (JAX-style).
    Adapted from provided code.
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def compute_lr(
    base_lr: float,
    lr_warmup_steps: int,
    lr_min_ratio: float,
    current_step: int,
    total_steps: int
) -> float:
    """Compute learning rate with warmup and cosine decay."""
    if current_step < lr_warmup_steps:
        return base_lr * float(current_step) / float(max(1, lr_warmup_steps))
    
    progress = float(current_step - lr_warmup_steps) / float(
        max(1, total_steps - lr_warmup_steps)
    )
    return base_lr * (
        lr_min_ratio + max(
            0.0,
            (1 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)),
        )
    )


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """Apply RMS normalization."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
