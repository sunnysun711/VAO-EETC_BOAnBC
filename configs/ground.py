# configs/ground.py
"""Ground configuration classes for VAO models."""

from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np

from configs.types_ import GROUNDS
from utils.ground_gen import generate_random_ground_points


@dataclass
class GroundConfigBase:
    """Base configuration class for ground and track design.
    
    Contains common parameters shared between M1 and M2 models.
    """
    # === ground / VAO params ===
    ground_name: str | GROUNDS = "random"
    ground_points: np.ndarray | None = None
    num_eval_points: int | None = None
    seed: int | None = None
    platform_intervals: int = 4
    ds: float = 50.0  # delta s (meters)
    
    # === geo constraint params ===
    i_max: float = 0.03      # max inclination
    di_max: float = 0.018    # max inclination difference
    ht_min: float = 13.0     # min height for tunnel
    hb_min: float = 10.0     # min height for bridge
    min_slope_length: int = 600  # min length for one slope
    
    # === scaling ===
    smooth_scale: float = 1.0
    
    # === derived attributes (set in __post_init__) ===
    S: int = field(init=False, default=0)
    e6g: List[float] = field(init=False, default_factory=list)
    sigma: int = field(init=False, default=0)
    y_range: Tuple[float, float] = field(init=False, default=(0.0, 0.0))
    
    def _load_ground_points(self) -> np.ndarray:
        """Load or generate ground points."""
        if self.ground_points is not None:
            # Use provided ground points
            gp = self.ground_points.copy()
            self.num_eval_points = gp.shape[0] - 1
            self.ds = float(gp[1, 0] - gp[0, 0])
            
        elif self.ground_name == "random":
            # Generate random ground
            assert self.num_eval_points is not None, "num_eval_points must be provided for random ground"
            gp = generate_random_ground_points(
                num_eval_points=self.num_eval_points,
                seed=self.seed,
                ds=self.ds,
                platform_intervals=self.platform_intervals,
                smooth_scale=self.smooth_scale,
                flat_grad_max=self.i_max,
            )
            
        else:
            # Load from CSV file
            gp = np.loadtxt(
                f"data/grounds/{self.ground_name}.csv", 
                delimiter=",", 
                skiprows=1
            )
            self.ds = float(gp[1, 0] - gp[0, 0])
            
            # Shift y to make sure the min height is 0
            y = gp[:, 1] - min(gp[:, 1])
            
            # Add platform intervals at start and end
            yl = [y[0]] * self.platform_intervals
            yr = [y[-1]] * self.platform_intervals
            
            gp = np.column_stack((
                np.arange(0, gp.shape[0] + self.platform_intervals * 2),
                np.concatenate((yl, y, yr))
            ))
        
            # Apply smooth scale
            gp[:, 1] *= self.smooth_scale
            
        return gp
    
    def _init_derived_attributes(self, ground_points: np.ndarray) -> None:
        """Initialize derived attributes from ground points."""
        self.ground_points = ground_points
        self.S = ground_points.shape[0] - 1
        self.e6g = list(ground_points[:, 1])
        self.sigma = int(np.ceil(self.min_slope_length / self.ds))
        self.y_range = (min(self.e6g), max(self.e6g))


@dataclass
class GroundConfigM2(GroundConfigBase):
    """Configuration for VAO-M2 model (Yang Dongying AUTCOM setups).
    
    Uses quadratic cost functions with SOC constraints.
    """
    # Override default geo constraints for M2
    i_max: float = 0.015
    di_max: float = 0.015
    ht_min: float = 8.0
    hb_min: float = 12.0
    min_slope_length: int = 900
    
    # === cost params (Yang Dongying AUTCOM params) ===
    # Earthwork parameters
    Me: float = 1.5          # arctangent of subgrade slope
    We: float = 12.0         # meters, width of subgrade on track level
    U_surface: float = 14.0  # $ for track surface / m
    U_cut: float = 7.0       # $ for earth cut / m^3
    U_fill: float = 4.2      # $ for earth fill / m^3
    
    # Tunnel and bridge parameters
    c_tn: float = 14310.0    # $ for tunnel / m
    c_bg: float = 2980.0     # $ for bridge / m
    c_pl: float = 2135.0     # $ for bridge piles
    r_b: float = 0.05        # pile cost growth rate
    L_span: float = 32.0     # meters, standard beam span length
    c6e_tn: float = 86000.0  # $ for tunnel entrance
    
    # === scaling ===
    obj_scale: float = 7.0   # from dollar to yen
    
    def __post_init__(self):
        ground_points = self._load_ground_points()
        self._init_derived_attributes(ground_points)
        

if __name__ == "__main__":
    gcfg = GroundConfigM2(ground_name="HP3", )
    print(gcfg.S)
    pass