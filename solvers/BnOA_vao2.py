# solvers/BnOA_vao2.py
"""Branch and Outer Approximation (BnOA) Solver for VAO2 Model."""

from typing import Optional
from dataclasses import dataclass, field
import time

import gurobipy as gp
from gurobipy import GRB

from configs import GroundConfigM2
from solvers.base import Solution, SolverBase
from solvers.greedy_vao import generate_greedy_track
from utils.gurobi import (
    STATUS_NAMES, extract_gp_solution, extract_gp_cb_solution,
    set_warm_start_vars
)

from models.vao2 import (
    COST_SCALE,
    rg_all,
    build_vao_oa_master_model,
    generate_flat_track,
    check_soc_violation,
    generate_single_oa_cut,
    add_oa_cut_lazy,
    get_vao_objective,
)

OA_CUT_MIN_VIOLATION = 1e-6
OA_CUT_MIN_H = 1e-6


def _provide_upper_bound_solution(
    model: gp.Model,
    solver: "VAO2OASolver",
    current_sol: dict,
    max_violation: float
):
    """Provide SOC-adjusted upper bound solution via cbSetSolution."""
    all_vars = solver.vao_v
    stats = solver.stats
    cfg = solver.gcfg

    adjusted_sol = {k: dict(v) if isinstance(v, dict) else v for k, v in current_sol.items()}

    soc_adjustments_made = False

    for s in rg_all(cfg.S):
        hf_val = current_sol["hf"][s]
        hf2_val = current_sol["hf2"][s]
        hf2_required = hf_val ** 2
        if hf2_required > hf2_val + 1e-6:
            adjusted_sol["hf2"][s] = hf2_required
            soc_adjustments_made = True

        hc_val = current_sol["hc"][s]
        hc2_val = current_sol["hc2"][s]
        hc2_required = hc_val ** 2
        if hc2_required > hc2_val + 1e-6:
            adjusted_sol["hc2"][s] = hc2_required
            soc_adjustments_made = True

        hb_val = current_sol["hb"][s]
        hb2_val = current_sol["hb2"][s]
        hb2_required = hb_val ** 2
        if hb2_required > hb2_val + 1e-6:
            adjusted_sol["hb2"][s] = hb2_required
            soc_adjustments_made = True

    if soc_adjustments_made:
        ds = cfg.ds
        U_surface = cfg.U_surface
        U_cut, U_fill = cfg.U_cut, cfg.U_fill
        We, Me = cfg.We, cfg.Me
        c_pl, L_span = cfg.c_pl, cfg.L_span
        r_b = cfg.r_b

        for s in rg_all(cfg.S):
            z2_val = current_sol["z2"][s]
            hf_val = adjusted_sol["hf"][s]
            hc_val = adjusted_sol["hc"][s]
            hf2_val = adjusted_sol["hf2"][s]
            hc2_val = adjusted_sol["hc2"][s]

            Ce_new = (U_surface * z2_val +
                     U_cut * (We * hc_val + Me * hc2_val) +
                     U_fill * (We * hf_val + Me * hf2_val)) * ds / COST_SCALE
            adjusted_sol["Ce"][s] = Ce_new

            hb_val = adjusted_sol["hb"][s]
            hb2_val = adjusted_sol["hb2"][s]
            Cp_new = c_pl * (hb_val + r_b * hb2_val) * ds / L_span / COST_SCALE
            adjusted_sol["Cp"][s] = Cp_new

    for var_name, var_dict in all_vars.items():
        if isinstance(var_dict, dict):
            sol_values = adjusted_sol[var_name]
            if isinstance(sol_values, dict):
                for key, var in var_dict.items():
                    model.cbSetSolution(var, sol_values[key])
        else:
            sol_value = adjusted_sol[var_name]
            if not isinstance(sol_value, dict):
                model.cbSetSolution(var_dict, sol_value)

    model.cbUseSolution()
    stats.n_ub_solutions_provided += 1

    proposed_obj: float = get_vao_objective(adjusted_sol, cfg.S, cfg.obj_scale)  # type: ignore

    if soc_adjustments_made:
        solver._log(
            f"  -> Provided UB solution (SOC-adjusted): obj={proposed_obj:.4f} "
            f"(max_SOC_viol={max_violation:.4e})"
        )
    else:
        solver._log(f"  -> Provided UB solution: obj={proposed_obj:.4f}")


@dataclass
class OASolverStats:
    """OA solver statistics."""
    n_oa_cuts: int = 0
    n_feas_cuts: int = 0
    n_nogood_cuts: int = 0
    n_callbacks: int = 0
    n_soc_violations: int = 0
    n_ub_solutions_provided: int = 0
    best_obj: float = float('inf')
    best_solution: dict = field(default_factory=dict)
    solve_time: float = 0.0

    def reset(self):
        self.n_oa_cuts = 0
        self.n_feas_cuts = 0
        self.n_nogood_cuts = 0
        self.n_callbacks = 0
        self.n_soc_violations = 0
        self.n_ub_solutions_provided = 0
        self.best_obj = float('inf')
        self.best_solution = {}
        self.solve_time = 0.0


def oa_callback(model: gp.Model, where: int):
    """OA callback: check SOC feasibility at integer solutions and add OA cuts."""
    if where != GRB.Callback.MIPSOL:
        return

    cfg = model._cfg
    all_vars = model._all_vars
    solver: VAO2OASolver = model._solver
    stats: OASolverStats = solver.stats

    stats.n_callbacks += 1
    S = cfg.S

    current_sol = extract_gp_cb_solution(all_vars, model)

    h_vars = {
        "hf": all_vars["hf"],
        "hc": all_vars["hc"],
        "hb": all_vars["hb"],
    }
    t_vars = {
        "hf2": all_vars["hf2"],
        "hc2": all_vars["hc2"],
        "hb2": all_vars["hb2"],
    }

    soc_pairs = [
        ("hf", "hf2", "OA_Fill"),
        ("hc", "hc2", "OA_Cut"),
        ("hb", "hb2", "OA_Pile"),
    ]

    cuts_added_this_iter = 0
    total_violation = 0.0
    max_violation = 0.0

    for h_name, t_name, cut_prefix in soc_pairs:
        for s in rg_all(S):
            h_val = current_sol[h_name][s]
            t_val = current_sol[t_name][s]

            violation = check_soc_violation(h_val, t_val)

            if violation > OA_CUT_MIN_VIOLATION:
                stats.n_soc_violations += 1
                total_violation += violation
                max_violation = max(max_violation, violation)

                cut = generate_single_oa_cut(
                    h_vars[h_name][s],
                    t_vars[t_name][s],
                    h_val,
                    t_val,
                    f"{cut_prefix}_{s}",
                    OA_CUT_MIN_H
                )

                if cut is not None:
                    add_oa_cut_lazy(model, cut)
                    stats.n_oa_cuts += 1
                    cuts_added_this_iter += 1

    if cuts_added_this_iter > 0:
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solver._log(f"Callback #{stats.n_callbacks}: Added {cuts_added_this_iter} OA cuts "
                   f"(max_viol={max_violation:.2e}, obj={current_obj:.2f})")
        _provide_upper_bound_solution(model, solver, current_sol, max_violation)

    if cuts_added_this_iter == 0:
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if current_obj < stats.best_obj:
            stats.best_obj = current_obj
            stats.best_solution = current_sol.copy()
            solver._log(f"  -> New best feasible solution: obj={current_obj:.4f}")


class VAO2OASolver(SolverBase):
    """Outer Approximation solver for the VAO2 problem."""

    def __init__(
        self,
        cfg,
        time_limit: float = 3600.0,
        mip_gap: float = 0.01,
        warm_start: bool = True,
        root_dir: str = "runs",
        name: str = "BnOA_vao2",
        verbose: bool = True,
        add_initial_oa_cuts: bool = True,
    ):
        super().__init__(
            cfg, None, time_limit, mip_gap, warm_start, root_dir, name, verbose
        )
        self.gcfg = cfg
        self.add_initial_oa_cuts = add_initial_oa_cuts

        self.vao_m: Optional[gp.Model] = None
        self.vao_v: dict[str, gp.tupledict] = {}
        self.vao_c: dict[str, gp.tupledict] = {}

        self.stats = OASolverStats()

    def build(self):
        """Build OA master problem (all linear constraints, no SOC)."""
        self._log("Building OA master problem...")

        self.vao_m, self.vao_v, self.vao_c = build_vao_oa_master_model(self.gcfg)

        if self.add_initial_oa_cuts:
            self._add_initial_cuts()

        self._log(f"Master problem built: {self.vao_m.NumVars} vars, "
                 f"{self.vao_m.NumConstrs} constrs")

        if self.warm_start:
            try:
                ws_sol = generate_greedy_track(self.gcfg)
                set_warm_start_vars(self.vao_v, ws_sol, set_continuous=True, debug_on=False)
                self._log("Warm start solution loaded.")
            except Exception as e:
                self._log(f"Warning: Failed to generate warm start: {e}")

    def _add_initial_cuts(self):
        """Add initial OA cuts at reference points to tighten the relaxation."""
        assert self.vao_m is not None, "Build master problem first"

        S = self.gcfg.S
        n_initial_cuts = 0

        Hb = self.gcfg.hb_min
        Ht = self.gcfg.ht_min
        H_max = self.gcfg.y_range[1] - self.gcfg.y_range[0]

        hf = self.vao_v["hf"]
        hc = self.vao_v["hc"]
        hb = self.vao_v["hb"]
        hf2 = self.vao_v["hf2"]
        hc2 = self.vao_v["hc2"]
        hb2 = self.vao_v["hb2"]

        for s in rg_all(S):
            ref_hf = Hb / 2
            if ref_hf > OA_CUT_MIN_H:
                self.vao_m.addConstr(
                    2 * ref_hf * hf[s] - hf2[s] <= ref_hf ** 2,
                    name=f"InitOA_Fill_{s}"
                )
                n_initial_cuts += 1

            ref_hc = Ht / 2
            if ref_hc > OA_CUT_MIN_H:
                self.vao_m.addConstr(
                    2 * ref_hc * hc[s] - hc2[s] <= ref_hc ** 2,
                    name=f"InitOA_Cut_{s}"
                )
                n_initial_cuts += 1

            ref_hb = H_max / 2
            if ref_hb > OA_CUT_MIN_H:
                self.vao_m.addConstr(
                    2 * ref_hb * hb[s] - hb2[s] <= ref_hb ** 2,
                    name=f"InitOA_Pile_{s}"
                )
                n_initial_cuts += 1

        self.vao_m.update()
        self._log(f"Added {n_initial_cuts} initial OA cuts")

    def solve(self):
        """Solve the VAO2 problem using OA."""
        if self.vao_m is None:
            raise RuntimeError("Model not built. Call build() first.")

        self.stats.reset()

        self.vao_m.Params.LogFile = str(self.run_dir / "solve.log")
        self.vao_m.Params.OutputFlag = 1
        self.vao_m.Params.LazyConstraints = 1
        self.vao_m.Params.TimeLimit = self.time_limit
        self.vao_m.Params.MIPGap = self.mip_gap

        self.vao_m._cfg = self.gcfg
        self.vao_m._all_vars = self.vao_v
        self.vao_m._solver = self

        self._log("=" * 60)
        self._log("Starting Branch-and-Outer-Approximation optimization...")
        self._log("=" * 60)

        start_time = time.time()
        self.vao_m.optimize(oa_callback)
        self.stats.solve_time = time.time() - start_time

        self._log("=" * 60)
        self._log(f"Optimization finished in {self.stats.solve_time:.2f} seconds")
        self._log("=" * 60)

        if self.vao_m.SolCount > 0:
            self._verify_and_store_solution()

        return

    def _verify_and_store_solution(self):
        """Verify and store the final solution."""
        assert self.vao_m is not None, "Build master problem first"
        assert self.vao_m.SolCount > 0, "No solution found"
        self._log("Verifying final solution...")

        final_sol = extract_gp_solution(self.vao_v)

        S = self.gcfg.S
        max_violation = 0.0

        soc_pairs = [("hf", "hf2"), ("hc", "hc2"), ("hb", "hb2")]

        for h_name, t_name in soc_pairs:
            for s in rg_all(S):
                h_val = final_sol[h_name][s]
                t_val = final_sol[t_name][s]
                violation = check_soc_violation(h_val, t_val)
                max_violation = max(max_violation, violation)

        if max_violation > 1e-4:
            self._log(f"[Warning] Final solution has SOC violation: {max_violation:.2e}")
        else:
            self._log(f"Final solution is SOC feasible (max violation: {max_violation:.2e})")

        obj = self.vao_m.ObjVal
        if obj < self.stats.best_obj or not self.stats.best_solution:
            self.stats.best_obj = obj
            self.stats.best_solution = final_sol

    def report(self):
        """Print solver report."""
        self._log("\n" + "=" * 60)
        self._log("       VAO2 Outer Approximation Solver Report")
        self._log("=" * 60)

        if self.vao_m is None:
            self._log("Model not built yet.")
            return

        status_name = STATUS_NAMES.get(self.vao_m.Status, str(self.vao_m.Status))
        self._log(f"Status: {status_name}")
        self._log(f"Solve time: {self.stats.solve_time:.2f} seconds")
        self._log(f"\nCut Statistics:")
        self._log(f"  - OA cuts added: {self.stats.n_oa_cuts}")
        self._log(f"  - Feasibility cuts added: {self.stats.n_feas_cuts}")
        self._log(f"  - No-good cuts added: {self.stats.n_nogood_cuts}")
        self._log(f"  - Total callbacks: {self.stats.n_callbacks}")
        self._log(f"  - Total SOC violations detected: {self.stats.n_soc_violations}")
        self._log(f"  - Upper bound solutions provided: {self.stats.n_ub_solutions_provided}")

        if self.vao_m.SolCount > 0:
            self._log(f"\nObjective Value: {self.stats.best_obj:.4f}")

            if self.stats.best_solution:
                z1_count = sum(1 for s, v in self.stats.best_solution.get("z1", {}).items() if v > 0.5)
                z2_count = sum(1 for s, v in self.stats.best_solution.get("z2", {}).items() if v > 0.5)
                z3_count = sum(1 for s, v in self.stats.best_solution.get("z3", {}).items() if v > 0.5)
                pi_count = sum(1 for s, v in self.stats.best_solution.get("pi", {}).items() if v > 0.5)

                self._log(f"\nTrack Type Distribution:")
                self._log(f"  - Tunnel segments (z1): {z1_count}")
                self._log(f"  - Earthwork segments (z2): {z2_count}")
                self._log(f"  - Bridge segments (z3): {z3_count}")
                self._log(f"  - VPI points (pi): {pi_count}")

        self._log("=" * 60 + "\n")

    def get_solution(self) -> Optional[Solution]:
        return Solution(vao_sol=self.stats.best_solution, SPs_sol=None)

    def get_objective(self) -> float:
        return self.stats.best_obj


if __name__ == "__main__":
    gcfg = GroundConfigM2(
        num_eval_points=150, ds=50, smooth_scale=0.8,
    )

    solver = VAO2OASolver(
        gcfg,
        time_limit=120,
        mip_gap=0.0,
        warm_start=True,
        verbose=False,
    )
    solver.build()
    solver.solve()
    solver.report()
