# solvers/base.py
"""Base classes and utilities for decomposition solvers."""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from io import TextIOWrapper
import json
from pathlib import Path
import pickle
from typing import Any, Optional

from configs.ground import GroundConfigBase
from configs.train import TrainConfig, TRAIN_TYPES


@dataclass
class TrainInstance:
    """Represents a single train instance (type, direction, scale).

    Attributes:
        name: Identifier for this instance
        train_config: Configuration for the train
        is_uphill: Whether the train travels uphill
        scale: Energy cost scaling factor
        is_time_constrained: bool, default is True
    """
    name: str
    train_config: TrainConfig
    is_uphill: bool
    scale: float
    is_time_constrained: bool = True


@dataclass
class Solution:
    """Solution container for VAO-TC decomposition problems.

    Stores optimization results from the VAO (Vertical Alignment Optimization) 
    master problem and TC (Train Control) subproblems.

    Attributes:
        vao_sol: Variable solutions for VAO model. 
            Keys are variable names (e.g., 'e', 'z1', 'z2'), values are {index: value} dicts.
        SPs_sol: Variable solutions for TC subproblems, keyed by train instance index.
            None if no subproblem solutions exist.

    Example:
        >>> sol = Solution(vao_sol={'e': {1: 10.5, 2: 11.2}}, SPs_sol={0: {'v': {1: 5.0}}})
        >>> sol.save("solution.pkl")
        >>> loaded = Solution.load("solution.pkl")
    
    Note:
        - Use save/load (pickle) for Python workflows - preserves int keys exactly.
        - Use to_json/from_json for human-readable output - int keys become strings.
    """
    vao_sol: dict[str, Any]
    SPs_sol: dict[int, dict[str, Any]] | None
    
    def save(self, filepath: str | Path) -> None:
        """Save solution to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> "Solution":
        """Load solution from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def to_json(self, filepath: str | Path) -> None:
        """Save solution to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({"vao_sol": self.vao_sol, "SPs_sol": self.SPs_sol}, f, indent=2)

    @classmethod
    def from_json(cls, filepath: str | Path) -> "Solution":
        """Load solution from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    


def build_train_instances(
    gcfg: GroundConfigBase, 
    vao_init_sol: dict[str, Any], 
    ti_info: list[tuple[TRAIN_TYPES, tuple[bool, ...], tuple[int, ...], tuple[float | None, ...]]],
    safety_margin: float = 1.3, 
    years: int = 50,
) -> list[TrainInstance]:
    """Build train instances from configuration.

    Args:
        gcfg: Ground configuration
        vao_init_sol: Initial solution for VAO variables
        ti_info: List of train type, directions, daily trips, and running_time (optional)
            e.g. [("CRH380AL_", (True, False), (100, 120), (2105, 2092)), ("HXD1D_", (False, True), (30, 40), (2736, 2712))]
        safety_margin: Safety margin for travel time
        years: Number of years to simulate
    """
    from models.tc_socp import simulate_sotc
    instances: list[TrainInstance] = []
    
    counter = 1
    for train_type, dirs, daily_trips, running_time in ti_info:
        for is_uphill_dir, daily_trips_dir, running_time_dir in zip(dirs, daily_trips, running_time):
            scale_dir = daily_trips_dir * 365 * years / (1000 * 3600)
            tcfg = TrainConfig(train_type=train_type)
            if running_time_dir is None:
                T_min_dir = simulate_sotc(
                    cfg=tcfg, vao_vars=vao_init_sol, S=gcfg.S, ds=gcfg.ds, is_uphill_dir=is_uphill_dir)
            else:
                T_min_dir = running_time_dir
            # print(f"{train_type} {is_uphill_dir} {T_min_dir:.2f}")
            tcfg.set_travel_time(custom_time=T_min_dir * safety_margin)
            instances.append(
                TrainInstance(train_config=tcfg, is_uphill=is_uphill_dir, scale=scale_dir, name=f"T{counter:02d}")
            )
            counter += 1

    return instances


class SolverBase:
    """Abstract base class for VAO-TC decomposition solvers.

    Provides common interface and output management for different solver 
    implementations (e.g., Benders decomposition, integrated solving).

    Attributes:
        run_dir: Directory path for solver outputs (solutions, logs, etc.)
        name: Identifier for this solver run

    Subclasses must implement:
        - build(): Construct the optimization model(s)
        - solve(): Execute the solving algorithm
        - get_solution(): Return Solution object or None if unsolved/infeasible

    Example:
        >>> class MySolver(SolverBase):
        ...     def __init__(self, root_dir: str = "runs", name: str = ""):
        ...         super().__init__(root_dir, name)
        ...     def build(self): ...
        ...     def solve(self): ...
        ...     def get_solution(self): ...
        >>> solver = MySolver(root_dir="runs", name="experiment_1")
        >>> solver.build()
        >>> solver.solve()
        >>> solver.save_solution()  # saves to runs/experiment_1_<timestamp>/
    """
    def __init__(
        self, 
        gcfg: GroundConfigBase,
        train_instances: Optional[list[TrainInstance]] = None,
        time_limit: float = 3600.0,
        mip_gap: float = 0.01,
        warm_start: bool = True,
        root_dir: str = "runs", 
        name: str = "", 
        verbose:bool=True,
    ):
        self.gcfg = gcfg
        self.train_instances = train_instances
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.warm_start = warm_start
        
        self.run_dir = create_run_dir(root_dir, name)
        self.name = name
        self.verbose = verbose
        self.log_header = f"[{self.name}] "
        self.log_f: TextIOWrapper = open(self.run_dir / "log.txt", "a", encoding="utf-8")
    
    def _log(self, msg: str, use_header: bool = True, print_msg: bool = True, end: str = "\n"):
        """Print and log message with solver name prefix."""
        header = self.log_header if use_header else ""
        self.log_f.write(f"{header}{msg}{end}")
        if end == "\n":
            self.log_f.flush()
        if self.verbose and print_msg:
            print(f"{header}{msg}", end=end)
    
    @abstractmethod
    def build(self):
        """Build the solver model."""
        pass
    
    @abstractmethod
    def solve(self):
        """Solve the model."""
        pass
    
    @abstractmethod
    def get_solution(self) -> Optional[Solution]:
        """Retrieve the solution after solving."""
        pass
    
    def save_solution(self):
        """Save the solution to disk."""
        sol = self.get_solution()
        if sol is None:
            print("[WARN] No solution to save")
            return
        sol.save(self.run_dir / "solution.pkl")
        sol.to_json(self.run_dir / "solution.json")


def create_run_dir(base: str = "runs", name: str = "") -> Path:
    """Create a timestamped directory for solver outputs.

    Args:
        base: Base directory name
        name: Name prefix for the run directory

    Returns:
        Path to the created directory
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir