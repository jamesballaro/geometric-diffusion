from typing import Dict, Any
from dataclasses import dataclass, field
import torch

from geodesic import SphericalCubicSpline
from score import ScoreProcessor

@dataclass
class BVPConfig:
    """Configuration for BVP Algorithm"""
    # Test setting
    device: torch.device

    test_name: str
    image_path1: str
    image_path2: str
    prompt1: str
    prompt2: str
    output_dir: str

    # CFG
    uncond_prompt: str
    neg_prompt: str
    noise_level: float
    alpha: float
    guidance_scale: float
    use_neg_cfg: bool

    # Output settings
    output_start_images: bool
    resolution: int
    num_output_imgs: int
    use_pu_sampling: bool
    grad_analysis_out: bool
    output_interval: int
    output_separate_images: bool = False
    project_to_sphere: bool = True

    # Grouped args
    grad_args: Dict[str, Any] = None
    bvp_opt_args: Dict[str, Any] = None
    bisection_args: Dict[str, Any] = None
    text_inv_args: Dict[str, Any] = None
    semantic_edit_args: Dict[str, Any] = None

@dataclass
class BVPState:
    """Mutable state shared during BVP optimization."""

    iter: int = 0

    image_latent1: torch.Tensor | None = None
    image_latent2: torch.Tensor | None = None
    image_tensor1: torch.Tensor | None = None
    image_tensor2: torch.Tensor | None = None

    prompt_embed_opt1: torch.Tensor | None = None
    prompt_embed_opt2: torch.Tensor | None = None

    uncond_prompt_embed: torch.Tensor | None = None
    neg_prompt_embed: torch.Tensor | None = None
    timesteps_out: torch.Tensor | None = None

    spline: "SphericalCubicSpline" | None = None

    score_unit: "ScoreProcessor" | None = None

    # Semantic edit state
    edit_idx: int = 0
