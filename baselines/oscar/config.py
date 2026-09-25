from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class DiversityConfig:
    num_steps: int = 30
    t_start: float = 1.0
    t_end: float = 0.0
    tau: float = 1.0
    eps_logdet: float = 1e-3
    feature_center: bool = True
    feature_l2norm: bool = True
    whiten: bool = True
    whiten_min_B: int = 16
    gamma0: float = 0.15
    gamma_max_ratio: float = 0.3
    partial_ortho: float = 0.5
    t_gate: Tuple[float,float] = (0.2, 0.9)
    sched_shape: str = "sin2"
    update_every: int = 1
    clip_image_size: int = 224
    angle_gate_deg: Optional[float] = None
    leverage_alpha: float = 0.5
    device: Optional[torch.device] = None
    noise_beta0: float = 0.0                     
    noise_use_same_gate: bool = True             
    noise_t_gate: Tuple[float,float] = (0.0, 0.7)
