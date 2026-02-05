# configs/train.py
"""Train configuration class."""

import json
from dataclasses import dataclass

from configs.types_ import TRAIN_TYPES, TRAVEL_TIME_SETUP_METHOD


@dataclass
class TrainConfig:
    """Configuration class for the train."""

    train_type: TRAIN_TYPES
    i_max: float = 0.03
    data_file_loc: str = "data/trains/"
    g: float = 9.81  # m/s^2
    min_v: float = 1e-1  # epsilon for velocity linearization
    
    # Attributes set in __post_init__
    data: dict | None = None
    mass: float = 0.0
    rho: float = 0.0
    max_tra_power: float = 0.0
    max_reg_bra_power: float = 0.0
    max_tra_force: float = 0.0
    max_reg_bra_force: float = 0.0
    max_pn_bra_force: float = 0.0
    max_acc: float = 0.0
    max_v: float = 0.0
    r0: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    rtn: float = 0.0
    eff_tra: float = 0.0
    eff_reg_bra: float = 0.0
    max_tra_cs: list = None  # type: ignore
    max_bra_cs: list = None  # type: ignore
    T: float = 0.0  # travel time, set by set_travel_time()

    def __post_init__(self):
        full_path = f"{self.data_file_loc}{self.train_type}.json"
        with open(full_path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)
        self.data = data

        self.mass = data["mass"]["value"]  # kg
        self.rho = data["rho"]["value"] / 100
        self.max_tra_power = data["max traction power"]["value"] * 1000  # W
        self.max_reg_bra_power = data["max reg braking power"]["value"] * 1000  # W
        self.max_tra_force = data["max traction force"]["value"] * 1000  # N
        self.max_reg_bra_force = data["max reg braking force"]["value"] * 1000  # N
        self.max_pn_bra_force = data["max pn braking force"]["value"] * 1000  # N
        self.max_acc = data["max deceleration"]["value"]  # m/s^2
        self.max_v = data["max speed"]["value"] / 3.6  # m/s
        self.r0 = data["rolling resistance r0"]["value"] * 1000  # N
        self.r1 = data["rolling resistance r1"]["value"] * 1000 * 3.6  # N/(m/s)
        self.r2 = data["rolling resistance r2"]["value"] * 1000 * 3.6**2  # N/(m/s)^2
        self.rtn = data["tunnel resistance rtn"]["value"] * 1000 * 3.6**2  # N/(m/s)^2
        self.eff_tra = data["efficiency traction"]["value"] / 100
        self.eff_reg_bra = data["efficiency reg brake"]["value"] / 100

        self.max_tra_cs = data.get("max traction constraint formula", [])
        self.max_bra_cs = data.get("max braking constraint formula", [])

    def estimate_travel_time(
        self,
        method: TRAVEL_TIME_SETUP_METHOD = "detailed",
        safety_margin: float = 1.2,
        total_distance: float = 0.0,
    ) -> float:
        """Estimate section running times (seconds).

        Args:
            method: Travel time setup method
                - simple: assuming average speed to be 0.7 * max_v
                - detailed: detailed model with friction, grade, and tunnel resistance
                - conservative: conservative model with higher safety margin
            safety_margin: Safety margin factor (default: 1.2)
            total_distance: Total distance (m) to be traveled

        Returns:
            Estimated running time in seconds
        """
        if method == "simple":
            avg_speed = self.max_v * 0.7
            t_estimated = total_distance / avg_speed

        elif method == "detailed":
            v_avg_accel = self.max_v / 2
            R_avg = self.r0 + self.r1 * v_avg_accel + self.r2 * v_avg_accel**2
            F_net_avg = self.max_tra_force - R_avg
            a_avg = F_net_avg / self.mass
            a_avg = min(a_avg, self.max_acc * 0.2)

            # Calculate full acceleration time and distance
            t_accel_full = self.max_v / a_avg
            s_accel_full = 0.5 * a_avg * t_accel_full**2

            a_decel = min(self.max_acc * 0.2, 1.0)
            t_decel_full = self.max_v / a_decel
            s_decel_full = 0.5 * self.max_v * t_decel_full

            # Check if distance is sufficient to accelerate to max_v
            if s_accel_full + s_decel_full > total_distance:
                # Situation A: traction, brake (no cruise)
                ratio = a_decel / a_avg if a_avg > 0.01 else 1.0
                s_accel = total_distance * ratio / (1 + ratio)
                s_decel = total_distance - s_accel

                t_accel = (s_accel * 2 / a_avg) ** 0.5
                t_decel = (s_decel * 2 / a_decel) ** 0.5
                t_base = t_accel + t_decel
            else:
                # Situation B: traction, cruise, brake
                s_cruise = total_distance - s_accel_full - s_decel_full
                v_cruise = self.max_v * 0.9
                t_cruise = s_cruise / v_cruise if s_cruise > 0 else 0
                t_base = t_accel_full + t_cruise + t_decel_full

            t_estimated = t_base

        elif method == "conservative":
            avg_speed = self.max_v * 0.2
            t_base = total_distance / avg_speed
            t_estimated = t_base * 1.8

        else:
            raise ValueError(f"Unknown travel time setup method: {method}")

        return t_estimated * safety_margin

    def set_travel_time(
        self,
        method: TRAVEL_TIME_SETUP_METHOD = "detailed",
        safety_margin: float = 2.0,
        custom_time: float | None = None,
        total_distance: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """Set section running time.

        Args:
            method: Estimation method
            safety_margin: Safety margin factor
            custom_time: Custom time (if provided, ignores estimation)
            total_distance: Total distance (m)
            verbose: Whether to print details
        """
        if custom_time is not None:
            self.T = custom_time
        else:
            self.T = self.estimate_travel_time(method, safety_margin, total_distance)

        if verbose:
            print(f"Running time set to: {self.T:.2f} s ({self.T/60:.2f} min)")
            print(f"  Total distance: {total_distance / 1000:.2f} km")
            print(f"  Average speed: {total_distance / self.T * 3.6:.2f} km/h")
            print(f"  Max speed: {self.max_v * 3.6:.2f} km/h")
            if custom_time is None:
                print(f"  Method: {method}, Safety margin: {safety_margin}")
                
                
if __name__ == "__main__":
    tcfg = TrainConfig(train_type="CRH380AL_")
    print(tcfg)