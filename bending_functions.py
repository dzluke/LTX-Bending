from __future__ import annotations

import torch


def add_scalar(latent: torch.Tensor, value: float) -> torch.Tensor:
    """Add a scalar value to the latent."""
    return latent + value


def multiply_scalar(latent: torch.Tensor, factor: float) -> torch.Tensor:
    """Multiply the latent by a scalar factor."""
    return latent * factor


def invert(latent: torch.Tensor) -> torch.Tensor:
    """Invert latent values by negating them."""
    return -latent


def reflect(latent: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reflect latent values along a chosen tensor dimension."""
    return torch.flip(latent, dims=[dim])


def rotate(latent: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Rotate the latent spatial plane by 90-degree increments."""
    return torch.rot90(latent, k=k, dims=(-2, -1))


def exponential(latent: torch.Tensor) -> torch.Tensor:
    """Apply an element-wise exponential transform to the latent."""
    return torch.exp(latent)


def logarithm(latent: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply a numerically stable element-wise natural logarithm."""
    return torch.log(torch.clamp(latent, min=eps))


def power(latent: torch.Tensor, exponent: float) -> torch.Tensor:
    """Raise each latent element to the given exponent."""
    return torch.pow(latent, exponent)
