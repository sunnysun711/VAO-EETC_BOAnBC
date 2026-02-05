# solvers/BOAnBC_vao2_eetc.py
"""Branch Outer Approximation and Benders Cuts (BOAnBC) Solver.

Hybrid decomposition algorithm combining:
- Branch-and-Bound: Handle integer variables (z1, z2, z3, pi)
- Outer Approximation: Handle VAO's SOC constraints
- Benders Decomposition: Handle TC subproblems coupling through (e, z1)

Master Problem (MILP):
    Variables: VAO (z1, z2, z3, pi, e, hf, hc, hb, etc.) + θ_i (surrogate for TC energy)
    Constraints: VAO linear + OA cuts (lazy) + Benders cuts (lazy)

Subproblems (SOCP):
    For each train: TC model with fixed (e, z1) from master
"""

from typing import Any, Callable, Optional
from dataclasses import dataclass, field
import time

import gurobipy as gp
from gurobipy import GRB

from configs import GroundConfigM2, TrainConfig
from utils.gurobi import (
    STATUS_NAMES, 
    extract_gp_solution, 
    extract_gp_cb_solution,
    set_warm_start_vars,
    set_params_for_robust_solve_socp,
)

from models.vao2 import (
    add_oa_cut_to_model,
    compute_initial_oa_cuts,
    rg_all,
    add_mp_variables as add_vao_mp_variables,
    add_mp_constraints as add_vao_mp_constraints,
    add_sp_variables as add_vao_sp_variables,
    add_sp_constraints as add_vao_sp_constraints,
    get_vao_objective,
    generate_flat_track,
    check_soc_violation,
    generate_single_oa_cut,
    build_vao_model,
    COST_SCALE
)

from models.tc_socp import (
    add_tc_variables,
    add_tc_constraints,
    add_eetc_objective,
    solve_tc_subproblem,
    try_extract_optim_cut,
)

from solvers.base import TrainInstance, build_train_instances, SolverBase, Solution
from solvers.greedy_vao import generate_greedy_track

OA_CUT_MIN_VIOLATION = 1e-6
OA_CUT_MIN_H = 1e-6
TC_OPTIM_CUT_TOL = 1e-4  # Tolerance for adding TC optimality cuts

@dataclass
class BOAnBCSolverStats:
    """Statistics for BOAnBC solver."""
    n_vao_oa_cuts: int = 0
    n_vao_soc_violations: int = 0
    n_tc_optim_cuts: int = 0
    n_tc_cuts_socp: int = 0
    n_tc_cuts_slack_T: int = 0
    n_tc_cuts_slack_U: int = 0
    n_tc_cuts_fixed: int = 0
    n_tc_subproblems_solved: int = 0
    n_tc_subproblems_infeasible: int = 0
    n_tc_dual_failures: int = 0
    n_callbacks: int = 0
    n_integer_solutions: int = 0
    n_feasible_solutions: int = 0
    n_ub_solutions_provided: int = 0  # Number of upper bound solutions provided via cbSetSolution

    best_obj: float = float('inf')
    gurobi_obj: float = float('inf')
    best_vao_cost: float = float('inf')
    best_tc_cost: float = float('inf')
    best_theta_cost: float = float('inf')
    best_solution: dict = field(default_factory=dict)

    solve_time: float = 0.0
    tc_solve_time: float = 0.0
    log_info: str = ""

    def reset(self):
        """Reset all statistics."""
        self.n_vao_oa_cuts = 0
        self.n_vao_soc_violations = 0
        self.n_tc_optim_cuts = 0
        self.n_tc_cuts_socp = 0
        self.n_tc_cuts_slack_T = 0
        self.n_tc_cuts_slack_U = 0
        self.n_tc_cuts_fixed = 0
        self.n_tc_subproblems_solved = 0
        self.n_tc_subproblems_infeasible = 0
        self.n_tc_dual_failures = 0
        self.n_callbacks = 0
        self.n_integer_solutions = 0
        self.n_feasible_solutions = 0
        self.n_ub_solutions_provided = 0
        self.best_obj = float('inf')
        self.gurobi_obj = float('inf')
        self.best_vao_cost = float('inf')
        self.best_tc_cost = float('inf')
        self.best_theta_cost = float('inf')
        self.best_solution = {}
        self.solve_time = 0.0
        self.tc_solve_time = 0.0
        self.log_info = ""


def boanbc_callback(model: gp.Model, where: int):
    if where != GRB.Callback.MIPSOL:
        return

    solver: BOAnBCSolver = model._solver
    all_vars: dict = solver.vao_var
    stats: BOAnBCSolverStats = solver.stats

    stats.n_callbacks += 1
    stats.n_integer_solutions += 1

    current_sol = extract_gp_cb_solution(all_vars, model)

    # --- Part 1: VAO SOC Constraint Checking and OA Cut Generation ---
    vao_cuts_added = _add_vao_oa_cuts(model, solver, current_sol)

    if vao_cuts_added > 0:
        solver._log(f"Callback #{stats.n_callbacks}: Added {vao_cuts_added} VAO-OA cuts")

    # --- Part 2: TC Subproblem Solving and Benders Cut Generation ---
    tc_cuts_added, all_tc_satisfied, encountered_failure, tc_obj_values = _add_tc_benders_cuts(
        model, solver, current_sol
    )

    # --- Part 3: Update Upper Bound with cbSetSolution ---
    # If no OA cuts were added and we have all TC objective values, provide upper bound
    # Note: OA cuts don't affect feasibility, only objective
    # We can always use cbSetSolution as long as we have valid TC costs
    if not encountered_failure and all(tc_obj is not None for tc_obj in tc_obj_values):
        _provide_upper_bound_solution(model, solver, current_sol, tc_obj_values)

    # --- Part 4: Update Best Solution ---
    # Only if: (1) all θ values are tight, AND (2) no failures encountered
    if all_tc_satisfied and not encountered_failure:
        stats.n_feasible_solutions += 1
        _update_best_solution(model, solver, current_sol)
    elif tc_cuts_added > 0:
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solver._log(
            f"Callback #{stats.n_callbacks}: "
            f"TC optim cuts={tc_cuts_added}, obj={current_obj:.2f}"
        )
    elif encountered_failure:
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solver._log(
            f"Callback #{stats.n_callbacks}: "
            f"TC failure encountered, no valid cut available, obj={current_obj:.2f}"
        )


def _add_vao_oa_cuts(
    model: gp.Model, 
    solver: "BOAnBCSolver", 
    current_sol: dict
) -> int:
    gcfg = solver.gcfg
    all_vars = solver.vao_var
    stats = solver.stats
    S = gcfg.S
    
    h_vars = {"hf": all_vars["hf"], "hc": all_vars["hc"], "hb": all_vars["hb"]}
    t_vars = {"hf2": all_vars["hf2"], "hc2": all_vars["hc2"], "hb2": all_vars["hb2"]}
    
    soc_pairs = [
        ("hf", "hf2", "OA_Fill"),
        ("hc", "hc2", "OA_Cut"),
        ("hb", "hb2", "OA_Pile"),
    ]
    
    cuts_added = 0
    
    for h_name, t_name, cut_prefix in soc_pairs:
        for s in rg_all(S):
            h_val = current_sol[h_name][s]
            t_val = current_sol[t_name][s]
            
            violation = check_soc_violation(h_val, t_val)
            
            if violation > OA_CUT_MIN_VIOLATION:
                stats.n_vao_soc_violations += 1
                
                cut = generate_single_oa_cut(
                    h_vars[h_name][s],
                    t_vars[t_name][s],
                    h_val,
                    t_val,
                    f"{cut_prefix}_{s}",
                    OA_CUT_MIN_H
                )
                
                if cut is not None:
                    model.cbLazy(cut.get_expr() <= 0)
                    stats.n_vao_oa_cuts += 1
                    cuts_added += 1
    
    return cuts_added


def _add_tc_benders_cuts(
    model: gp.Model,
    solver: "BOAnBCSolver",
    vao_sol: dict
) -> tuple[int, bool, bool, list[float]]:
    gcfg = solver.gcfg
    train_instances = solver.train_instances
    all_vars = solver.vao_var
    stats = solver.stats
    log: Callable = solver._log
    run_dir = solver.run_dir
    tc_mods: list[gp.Model] = solver.tc_mods

    cuts_added = 0
    all_tc_satisfied = True
    encountered_failure = False
    tc_obj_values = [] 
    
    for idx, ti in enumerate(train_instances):
        theta_var = all_vars["theta"][idx]
        theta_val = model.cbGetSolution(theta_var)

        _start = time.time()
        tc_model, tc_obj = solve_tc_subproblem(
            tc_model=tc_mods[idx],
            tcfg=ti.train_config, is_uphill_dir=ti.is_uphill, gcfg=gcfg,
            vao_sol=vao_sol,
        )
        stats.tc_solve_time += time.time() - _start
        stats.n_tc_subproblems_solved += 1

        tc_obj_values.append(tc_obj)

        if tc_obj is not None and tc_obj - theta_val <= TC_OPTIM_CUT_TOL:
            continue  # feasible solution found, accept it

        all_tc_satisfied = False

        cut_expr, info = try_extract_optim_cut(
            tc_model=tc_model,
            mp_vars=all_vars,
            vao_sol=vao_sol,
            tcfg=ti.train_config,
            ti_idx=idx,
            ti_scale = ti.scale,
            gcfg=gcfg,
            is_uphill_dir=ti.is_uphill,
            run_dir=run_dir,
            n_callback=stats.n_callbacks,
            log_method=log,
            log_prefix=f"  EETC-{idx:02d} ({ti.name}): ",
        )

        if cut_expr is not None:  # dual extraction success: socp, fixed-lp, or active-constraint lp
            model.cbLazy(theta_var >= cut_expr)
            stats.n_tc_optim_cuts += 1
            cuts_added += 1

            # Track cut type breakdown
            if info == "socp":
                stats.n_tc_cuts_socp += 1
            elif "slack_T" in info:
                stats.n_tc_cuts_slack_T += 1
            elif "slack_U" in info:
                stats.n_tc_cuts_slack_U += 1
            elif info == "active" or info == "fixed":
                stats.n_tc_cuts_fixed += 1

            log(
                f"  EETC-{idx:02d} ({ti.name}): "
                f"θ={theta_val:.2f}, added optim cut ({info})."
            )
            continue

        stats.n_tc_dual_failures += 1
        if "slack" in info:  # even adding slack doesn't make the problem feasible
            stats.n_tc_subproblems_infeasible += 1

        encountered_failure = True

    return cuts_added, all_tc_satisfied, encountered_failure, tc_obj_values


def _provide_upper_bound_solution(
    model: gp.Model,
    solver: "BOAnBCSolver",
    current_sol: dict,
    tc_obj_values: list[float]
):
    all_vars = solver.vao_var
    stats = solver.stats
    gcfg = solver.gcfg

    adjusted_sol = {k: dict(v) if isinstance(v, dict) else v for k, v in current_sol.items()}

    soc_adjustments_made = False
    max_soc_violation = 0.0

    for s in rg_all(gcfg.S):
        # SOC for filling: hf^2 <= hf2
        hf_val = current_sol["hf"][s]
        hf2_val = current_sol["hf2"][s]
        hf2_required = hf_val ** 2
        if hf2_required > hf2_val + 1e-6:  # Violated
            violation = hf2_required - hf2_val
            max_soc_violation = max(max_soc_violation, violation)
            adjusted_sol["hf2"][s] = hf2_required
            soc_adjustments_made = True

        # SOC for cutting: hc^2 <= hc2
        hc_val = current_sol["hc"][s]
        hc2_val = current_sol["hc2"][s]
        hc2_required = hc_val ** 2
        if hc2_required > hc2_val + 1e-6:  # Violated
            violation = hc2_required - hc2_val
            max_soc_violation = max(max_soc_violation, violation)
            adjusted_sol["hc2"][s] = hc2_required
            soc_adjustments_made = True

        # SOC for pile: hb^2 <= hb2
        hb_val = current_sol["hb"][s]
        hb2_val = current_sol["hb2"][s]
        hb2_required = hb_val ** 2
        if hb2_required > hb2_val + 1e-6:  # Violated
            violation = hb2_required - hb2_val
            max_soc_violation = max(max_soc_violation, violation)
            adjusted_sol["hb2"][s] = hb2_required
            soc_adjustments_made = True

    # If SOC adjustments were made, recalculate Ce and Cp
    if soc_adjustments_made:
        ds = gcfg.ds
        U_surface = gcfg.U_surface
        U_cut, U_fill = gcfg.U_cut, gcfg.U_fill
        We, Me = gcfg.We, gcfg.Me
        c_pl, L_span = gcfg.c_pl, gcfg.L_span
        r_b = gcfg.r_b

        for s in rg_all(gcfg.S):
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

    # Set all variables to adjusted solution values
    for var_name, var_dict in all_vars.items():
        if var_name == "theta":
            for idx in range(len(tc_obj_values)):
                model.cbSetSolution(var_dict[idx], tc_obj_values[idx])
        else:
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

    vao_cost: float = get_vao_objective(adjusted_sol, gcfg.S, gcfg.obj_scale)  # type: ignore
    tc_cost_actual = sum(tc_obj_values)
    proposed_obj = vao_cost + tc_cost_actual
    theta_cost = sum(current_sol["theta"][idx] for idx in range(len(tc_obj_values)))
    theta_gap = tc_cost_actual - theta_cost

    if soc_adjustments_made:
        solver._log(
            f"  -> Provided UB solution (SOC-adjusted): obj={proposed_obj:.4f} "
            f"(VAO={vao_cost:.4f}, TC_actual={tc_cost_actual:.4f}, θ_gap={theta_gap:.4f}, "
            f"max_SOC_viol={max_soc_violation:.4e})"
        )
    else:
        solver._log(
            f"  -> Provided UB solution: obj={proposed_obj:.4f} "
            f"(VAO={vao_cost:.4f}, TC_actual={tc_cost_actual:.4f}, θ_gap={theta_gap:.4f})"
        )


def _update_best_solution(
    model: gp.Model,
    solver: "BOAnBCSolver",
    current_sol: dict
):
    stats = solver.stats
    gcfg = solver.gcfg
    S = gcfg.S

    vao_cost: float = get_vao_objective(current_sol, S, gcfg.obj_scale)  # type: ignore

    actual_tc_cost = 0.0
    theta_tc_cost = 0.0

    for idx, train_inst in enumerate(solver.train_instances):
        theta_val = current_sol["theta"][idx]
        theta_tc_cost += theta_val

        tc_model, tc_obj = solve_tc_subproblem(
            tc_model=solver.tc_mods[idx],
            tcfg=train_inst.train_config, is_uphill_dir=train_inst.is_uphill, gcfg=gcfg,
            vao_sol=current_sol,
        )

        if tc_model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] and tc_obj is not None:
            actual_tc_cost += tc_obj
        else:
            actual_tc_cost += theta_val
            solver._log(
                f"  [Warning] EETC-{idx:02d} ({train_inst.name}) "
                f"failed in _update_best_solution, using θ={theta_val:.4f}")

    true_obj = vao_cost + actual_tc_cost
    gurobi_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

    if true_obj < stats.best_obj:
        stats.best_obj = true_obj
        stats.gurobi_obj = gurobi_obj
        stats.best_solution = current_sol.copy()
        stats.best_vao_cost = vao_cost
        stats.best_tc_cost = actual_tc_cost
        stats.best_theta_cost = theta_tc_cost

        theta_gap = actual_tc_cost - theta_tc_cost
        solver._log(
            f"  -> New best feasible: true_obj={true_obj:.4f} "
            f"(VAO={vao_cost:.4f}, TC_actual={actual_tc_cost:.4f}, θ_gap={theta_gap:.4f})"
        )


# =============================================================================
# Main Solver Class
# =============================================================================

class BOAnBCSolver(SolverBase):
    def __init__(
        self,
        gcfg: GroundConfigM2,
        train_instances: list[TrainInstance],
        add_init_oa_cuts: bool = True,
        time_limit: float = 3600.0,
        mip_gap: float = 0.01,
        warm_start: bool = True,
        root_dir: str = "runs",
        name: str = "BOAnBC",
        verbose: bool = True,
    ):
        super().__init__(
            gcfg, train_instances, time_limit, mip_gap, warm_start, root_dir, name, verbose
        )
        self.gcfg: GroundConfigM2 = gcfg
        self.train_instances = train_instances
        self.add_init_oa_cuts = add_init_oa_cuts
        
        self.vao_mod: Optional[gp.Model] = None
        self.vao_var: dict[str, Any] = {}
        self.vao_cs: dict[str, Any] = {}
        
        self.tc_mods: list[gp.Model] = []
        
        self.stats = BOAnBCSolverStats()
    
    def build(self):
        """Build master problem."""
        self._log("Building BOAnBC master problem...")
        
        gcfg = self.gcfg
        
        self.vao_mod = gp.Model("boanbc_master")
        
        vao_mp_vars = add_vao_mp_variables(self.vao_mod, gcfg)
        vao_sp_vars = add_vao_sp_variables(self.vao_mod, gcfg)
        n_trains = len(self.train_instances)
        theta = self.vao_mod.addVars(n_trains, lb=0.0, name="theta")
        self.vao_var = {**vao_mp_vars, **vao_sp_vars, "theta": theta}
        
        vao_mp_cs = add_vao_mp_constraints(self.vao_mod, gcfg, vao_mp_vars)
        vao_sp_cs = add_vao_sp_constraints(
            self.vao_mod, gcfg, vao_sp_vars, vao_mp_vars,
            is_height_included=True,
            is_soc_included=False,
            is_q_cost_included=True,
            is_l_cost_included=False,
        )
        self.vao_cs = {**vao_mp_cs, **vao_sp_cs}
        self._add_initial_oa_cuts()
        
        vao_cost = get_vao_objective(self.vao_var, S=gcfg.S, scale=gcfg.obj_scale)
        tc_cost = gp.quicksum(theta[i] for i in range(n_trains))
        self.vao_mod.setObjective(vao_cost + tc_cost, GRB.MINIMIZE)
        self.vao_mod.update()
        
        self._log(
            f"Master problem built: "
            f"{self.vao_mod.NumVars} vars, {self.vao_mod.NumConstrs} constrs, "
            f"{n_trains} train instances"
        )
        
        self._log("Building BOAnBC EETC subproblems...")
        for ti in self.train_instances:
            tc_mod = gp.Model(f"tc_{ti.name}")
            tc_vars = add_tc_variables(tc_mod, ti.train_config, S=gcfg.S)
            tc_cs = add_tc_constraints(
                tc_mod, ti.train_config, None, tc_vars, 
                S=gcfg.S, ds=gcfg.ds, is_uphill_dir=ti.is_uphill, 
                is_time_constrained=ti.is_time_constrained, 
                is_coupling_included=False,
                is_soc_included=True
            )
            tc_obj = add_eetc_objective(
                tc_mod, ti.train_config, tc_vars, 
                S=gcfg.S, ds=gcfg.ds, scale=ti.scale,
                is_reg_included=False, 
            )
            tc_mod._variables = tc_vars
            tc_mod._constraints = tc_cs
            tc_mod._objective_expr = tc_obj
            tc_mod.Params.LogToConsole = 0
            set_params_for_robust_solve_socp(tc_mod)
            self.tc_mods.append(tc_mod)
        
        self._log(
            f"EETC subproblems built: "
            f"{sum(tc_mod.NumVars for tc_mod in self.tc_mods)} vars, "
            f"{sum(tc_mod.NumConstrs for tc_mod in self.tc_mods)} constrs, "
            f"{n_trains} train instances"
        )
    
    def _add_initial_oa_cuts(self):
        if self.vao_mod is None or not self.add_init_oa_cuts:
            return
        
        cuts = compute_initial_oa_cuts(self.vao_var, self.gcfg, min_h_for_cut=OA_CUT_MIN_H)
        for cut in cuts:
            add_oa_cut_to_model(self.vao_mod, cut)
        self._log(f"Added {len(cuts)} initial OA cuts")
    
    def _generate_warm_start(self) -> dict:
        self._log("=" * 70)
        self._log("Generating warm start solution...")
        self._log("=" * 70)
        
        gcfg = self.gcfg
        
        self._log("Step 1: Generating initial track...")
        try:
            ws_sol = generate_greedy_track(gcfg)
            self._log("  Using greedy track as initial solution")
        except Exception as e:
            self._log(f"  Greedy track failed ({e}), using flat track")
            ws_sol = generate_flat_track(gcfg, is_mid_vpi=True)
        
        self._log("Step 2: Optimizing VAO problem...")
        vao_m, vao_v, vao_c = build_vao_model(gcfg, scale=gcfg.obj_scale, is_soc_included=True)
        set_warm_start_vars(vao_v, ws_sol, debug_on=False, set_continuous=True)
        hbar = dict(enumerate(gcfg.e6g, start=1))
        vao_m.setObjective(
            gp.quicksum((vao_v['e'][s] - hbar[s])**2 for s in rg_all(gcfg.S)), 
            gp.GRB.MINIMIZE
        )
        vao_m.Params.OutputFlag = 0
        vao_m.Params.TimeLimit = 10.0
        vao_m.Params.MIPGap = 0.02
        vao_m.Params.NumericFocus = 3
        vao_m.Params.FeasibilityTol = 1e-6
        vao_m.optimize()
        if vao_m.SolCount == 0:
            self._log(
                f"  [Warning] VAO optimization failed: "
                f"{STATUS_NAMES.get(vao_m.Status, vao_m.Status)}"
            )
            self._log("  Falling back to initial solution")
            ws_sol["theta"] = {i: 1e9 for i in range(len(self.train_instances))}
            self._log("=" * 70)
            return ws_sol
        vao_sol = extract_gp_solution(vao_v)
        vao_obj = vao_m.ObjVal
        self._log(f"  VAO optimization completed: obj={vao_obj:.4f}")
        
        self._log("Step 3: Solving TC subproblems to determine theta values...")
        theta_values = {}
        theta_margin = 1.02  # 2% margin for TC cost to avoid numerical instability
        tc_costs_actual = []
        tc_costs_with_margin = []
        for idx, ti in enumerate(self.train_instances):
            self._log(f"  Solving EETC-{idx:02d} ({ti.name})...")
            tc_model, tc_obj = solve_tc_subproblem(
                tc_model=self.tc_mods[idx], 
                tcfg=ti.train_config, is_uphill_dir=ti.is_uphill, gcfg=gcfg, 
                vao_sol=vao_sol,
            )
            if tc_model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] and tc_obj is not None:
                tc_costs_actual.append(tc_obj)
                theta_with_margin = tc_obj * theta_margin
                theta_values[idx] = theta_with_margin
                tc_costs_with_margin.append(theta_with_margin)
                self._log(
                    f"    Status: {STATUS_NAMES.get(tc_model.Status, tc_model.Status)}, "
                    f"TC_obj: {tc_obj:.4f}, "
                    f"θ (with {(theta_margin-1)*100:.0f}% margin): {theta_with_margin:.4f}"
                )
            else:
                self._log(
                    f"    [Warning] TC subproblem failed: "
                    f"{STATUS_NAMES.get(tc_model.Status, tc_model.Status)}, "
                    f"Setting θ to large penalty value"
                )
                theta_values[idx] = 1e9
                tc_costs_actual.append(1e9)
                tc_costs_with_margin.append(1e9)
        
        self._log("Step 4: Setting warm start solution...")
        vao_sol["theta"] = theta_values
        
        self._log("Step 5: Summary of warm start solution...")
        vao_cost = get_vao_objective(vao_sol, S=gcfg.S, scale=gcfg.obj_scale)
        tc_cost_actual = sum(tc_costs_actual)
        tc_cost_with_margin = sum(tc_costs_with_margin)
        total_obj_actual = vao_cost + tc_cost_actual
        total_obj_with_margin = vao_cost + tc_cost_with_margin
        margin_diff = tc_cost_with_margin - tc_cost_actual
        
        self._log("=" * 70)
        self._log("Warm Start Summary:")
        self._log(f"  VAO construction cost:      {vao_cost:.4f}")
        self._log(f"  TC actual total:            {tc_cost_actual:.4f}")
        self._log(f"  TC with margin total:       {tc_cost_with_margin:.4f}")
        self._log(f"  Margin difference:          {margin_diff:.4f} ({margin_diff/tc_cost_actual*100:.1f}%)")
        self._log(f"  Total obj (actual):         {total_obj_actual:.4f}")
        self._log(f"  Total obj (with margin):    {total_obj_with_margin:.4f}")
        self._log(f"  Number of train instances:  {len(self.train_instances)}")
        self._log("=" * 70)
        self._log("Warm start solution successfully set!")
        self._log("=" * 70)
        
        return vao_sol
    
    def solve(self):
        if self.vao_mod is None:
            raise RuntimeError("Model not built. Call build() first.")
        
        self.stats.reset()
        
        self.vao_mod.Params.LogFile = str(self.run_dir / "solve.log")
        self.vao_mod.Params.OutputFlag = 1
        self.vao_mod.Params.LazyConstraints = 1
        self.vao_mod.Params.TimeLimit = self.time_limit
        self.vao_mod.Params.MIPGap = self.mip_gap
        self.vao_mod.Params.NumericFocus = 2
        self.vao_mod.Params.FeasibilityTol = 1e-4
        
        if self.warm_start:
            try:
                ws_sol = self._generate_warm_start()
                set_warm_start_vars(self.vao_var, ws_sol, debug_on=False, set_continuous=True)
            except Exception as e:
                self._log("=" * 70)
                self._log(f"[ERROR] Failed to generate warm start: {e}")
                import traceback
                self._log(f"Traceback:\n{traceback.format_exc()}")
                self._log("Proceeding without warm start")
                self._log("=" * 70)
        
        self.vao_mod._solver = self
        
        self._log("=" * 70)
        self._log("Starting BOAnBC optimization (VAO + EETC)...")
        self._log(f"  Time limit: {self.time_limit}s, MIP gap: {self.mip_gap*100:.1f}%")
        self._log("=" * 70)
        
        start_time = time.time()
        self.vao_mod.optimize(boanbc_callback)
        self.stats.solve_time = time.time() - start_time
        
        self._log("=" * 70)
        self._log(f"Optimization finished in {self.stats.solve_time:.2f} seconds")
        self._log("=" * 70)
        
        if self.vao_mod.SolCount > 0:
            self._verify_final_solution()
            
        self.report()
        
        return
    
    def _verify_final_solution(self):
        if self.vao_mod is None or self.vao_mod.SolCount == 0:
            return
        
        self._log("Verifying final solution...")
        
        final_sol = extract_gp_solution(self.vao_var)
        gcfg = self.gcfg
        
        max_vao_violation = 0.0
        for h_name, t_name in [("hf", "hf2"), ("hc", "hc2"), ("hb", "hb2")]:
            for s in rg_all(gcfg.S):
                h_val = final_sol[h_name][s]
                t_val = final_sol[t_name][s]
                violation = check_soc_violation(h_val, t_val)
                max_vao_violation = max(max_vao_violation, violation)
        
        if max_vao_violation > 1e-4:
            self._log(f"  [Warning] VAO SOC violation: {max_vao_violation:.2e}")
        else:
            self._log(f"VAO SOC feasible (max violation: {max_vao_violation:.2e})")
        
        vao_sol = {"e": final_sol["e"], "z1": final_sol["z1"]}
        max_tc_gap = 0.0
        actual_tc_costs = []
        theta_values = []
        final_sol['SP'] = {}
        
        for idx, ti in enumerate(self.train_instances):
            theta_val = final_sol["theta"][idx]
            theta_values.append(theta_val)
            
            tc_model, tc_obj = solve_tc_subproblem(
                tc_model=self.tc_mods[idx], 
                tcfg=ti.train_config, is_uphill_dir=ti.is_uphill, gcfg=gcfg, 
                vao_sol=vao_sol,
            )
            
            if tc_model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] and tc_obj is not None:
                actual_tc_costs.append(tc_obj)
                gap = tc_obj - theta_val
                max_tc_gap = max(max_tc_gap, abs(gap))
                final_sol['SP'][idx] = extract_gp_solution(tc_model._variables)  # add EETC solution to final_sol
                
                if gap > 1e-2:
                    self._log(
                        f"  [Warning] EETC-{idx:02d} ({ti.name}): "
                        f"θ={theta_val:.4f} < actual={tc_obj:.4f}, gap={gap:.4f}"
                    )
                elif gap < -1e-2:
                    self._log(
                        f"  [Warning] EETC-{idx:02d} ({ti.name}): "
                        f"θ={theta_val:.4f} > actual={tc_obj:.4f} (unexpected!)"
                    )
            elif tc_model.Status == GRB.INFEASIBLE:
                self._log(f"  [ERROR] EETC-{idx:02d} ({ti.name}): INFEASIBLE in verification!")
                actual_tc_costs.append(float('inf'))
            else:
                self._log(
                    f"  [ERROR] EETC-{idx:02d} ({ti.name}): "
                    f"Solver status {STATUS_NAMES.get(tc_model.Status, tc_model.Status)} in verification!"
                )
                actual_tc_costs.append(theta_val)
        
        if max_tc_gap <= 1e-2:
            self._log(f"TC objectives verified (max gap: {max_tc_gap:.2e})")
        
        # Compute costs
        vao_cost: float = get_vao_objective(final_sol, gcfg.S, gcfg.obj_scale)  # type: ignore
        actual_tc_total = sum(actual_tc_costs)
        theta_tc_total = sum(theta_values)
        true_obj = vao_cost + actual_tc_total
        gurobi_obj = self.vao_mod.ObjVal
        
        # Log comparison
        self._log(f"Gurobi reported obj: {gurobi_obj:.4f}")
        self._log(f"θ-based TC total: {theta_tc_total:.4f}")
        self._log(f"Actual TC total: {actual_tc_total:.4f}")
        self._log(f"True objective (VAO + actual TC): {true_obj:.4f}")
        
        if abs(gurobi_obj - true_obj) > 1e-1:
            self._log(f"  [Warning] Gurobi obj differs from true obj by {abs(gurobi_obj - true_obj):.4f}")
        
        # Update stats
        self.stats.best_obj = true_obj
        self.stats.gurobi_obj = gurobi_obj
        self.stats.best_solution = final_sol
        self.stats.best_vao_cost = vao_cost
        self.stats.best_tc_cost = actual_tc_total
        self.stats.best_theta_cost = theta_tc_total
    
    def report(self):
        report_str = ""
        
        report_str += "\n" + "=" * 70 + "\n"
        report_str += "         BOAnBC Solver Report (VAO + EETC)\n"
        report_str += "=" * 70 + "\n"
        
        if self.vao_mod is None:
            report_str += "Model not built yet.\n"
            self._log(report_str, False, False)
            print(report_str)
            return report_str
        
        status_name = STATUS_NAMES.get(self.vao_mod.Status, str(self.vao_mod.Status))
        report_str += f"Status: {status_name}\n"
        report_str += f"Solve time: {self.stats.solve_time:.2f} seconds\n"
        report_str += f"  - TC subproblem time: {self.stats.tc_solve_time:.2f} seconds\n"
        
        report_str += f"\nVAO OA Statistics:\n"
        report_str += f"  - OA cuts added: {self.stats.n_vao_oa_cuts}\n"
        report_str += f"  - SOC violations detected: {self.stats.n_vao_soc_violations}\n"
        
        report_str += f"\nTC Benders Statistics:\n"
        report_str += f"  - Optimality cuts added: {self.stats.n_tc_optim_cuts}\n"
        report_str += f"    - socp: {self.stats.n_tc_cuts_socp}\n"
        report_str += f"    - slack_T: {self.stats.n_tc_cuts_slack_T}\n"
        report_str += f"    - slack_U: {self.stats.n_tc_cuts_slack_U}\n"
        report_str += f"    - fixed: {self.stats.n_tc_cuts_fixed}\n"
        report_str += f"  - Subproblems solved: {self.stats.n_tc_subproblems_solved}\n"
        report_str += f"  - Subproblems infeasible: {self.stats.n_tc_subproblems_infeasible}\n"
        report_str += f"  - Dual extraction failures: {self.stats.n_tc_dual_failures}\n"
        
        report_str += f"\nCallback Statistics:\n"
        report_str += f"  - Total callbacks: {self.stats.n_callbacks}\n"
        report_str += f"  - Integer solutions checked: {self.stats.n_integer_solutions}\n"
        report_str += f"  - Fully feasible solutions: {self.stats.n_feasible_solutions}\n"
        report_str += f"  - Upper bound solutions provided: {self.stats.n_ub_solutions_provided}\n"
        
        if self.vao_mod.SolCount > 0:
            report_str += f"\nObjective Breakdown:\n"
            report_str += f"  - True Total (VAO + actual TC): {self.stats.best_obj:.4f}\n"
            report_str += f"  - Gurobi Reported (VAO + θ): {self.stats.gurobi_obj:.4f}\n"
            report_str += f"  - VAO construction cost: {self.stats.best_vao_cost:.4f}\n"
            report_str += f"  - TC energy (actual): {self.stats.best_tc_cost:.4f}\n"
            report_str += f"  - TC energy (θ-based): {self.stats.best_theta_cost:.4f}\n"
            
            theta_gap = self.stats.best_tc_cost - self.stats.best_theta_cost
            if abs(theta_gap) > 1e-2:
                report_str += f"  - [!] θ gap: {theta_gap:.4f}\n"
            
            if self.stats.best_solution:
                z1_count = sum(1 for v in self.stats.best_solution.get("z1", {}).values() if v > 0.5)
                z2_count = sum(1 for v in self.stats.best_solution.get("z2", {}).values() if v > 0.5)
                z3_count = sum(1 for v in self.stats.best_solution.get("z3", {}).values() if v > 0.5)
                pi_count = sum(1 for v in self.stats.best_solution.get("pi", {}).values() if v > 0.5)
                
                report_str += f"\nTrack Type Distribution:\n"
                report_str += f"  - Tunnel (z1): {z1_count}\n"
                report_str += f"  - Earthwork (z2): {z2_count}\n"
                report_str += f"  - Bridge (z3): {z3_count}\n"
                report_str += f"  - VPI points (pi): {pi_count}\n"
        
        report_str += "=" * 70 + "\n\n"
        
        self._log(report_str, False, False)
        print(report_str)
        return report_str
    
    def get_solution(self) -> Optional[Solution]:
        vao_sol = {k: v for k, v in self.stats.best_solution.items() if k != "SP"}
        SPs_sol = self.stats.best_solution["SP"]
        return Solution(vao_sol=vao_sol, SPs_sol=SPs_sol)
    
    def get_objective(self) -> float:
        return self.stats.best_obj


def solve_vao_eetc_boanbc(
    gcfg: GroundConfigM2,
    train_instances: list[TrainInstance],
    mip_gap: float = 0.01,
    time_limit: float = 3600.0,
    verbose: bool = True,
    warm_start: bool = True,
) -> BOAnBCSolver:
    solver = BOAnBCSolver(
        gcfg, train_instances, verbose=verbose, add_init_oa_cuts=True,
        time_limit=time_limit, mip_gap=mip_gap, warm_start=warm_start)
    solver.build()
    solver.solve()
    # solver.report()
    solver.save_solution()
    
    return solver


if __name__ == "__main__":
    # Create ground configuration
    # gcfg = GroundConfigM2(
    #     num_eval_points=120, 
    #     ds=50, 
    #     # smooth_scale=0.5,
    #     # seed=2058068
    # )
    # gcfg = GroundConfigM2("HP2")
    gcfg = GroundConfigM2("HP3")
    
    print("=" * 70)
    print(f"Testing BOAnBC Solver (VAO + EETC) on {gcfg.ground_name}")
    print("=" * 70)
    
    # Create train instances
    try:
        ws_sol = generate_greedy_track(gcfg)
    except Exception as e:
        ws_sol = generate_flat_track(gcfg, is_mid_vpi=True)
    train_instances = build_train_instances(
        gcfg, ws_sol, 
        # ti_info=[("NL_intercity_VIRM6_", (True, False), (120, 120), (855, 850))],
        # safety_margin=1.15, 
        ti_info=[("NL_intercity_VIRM6_", (True, False), (120, 120), (1093, 1088))],
        safety_margin=1.15, 
        # ti_info=[("CRH380AL_", (True, False), (50, 50), (2105, 2092)), ("HXD1D_", (False, True), (8, 8), (2736, 2712))],
        # safety_margin=1.2, 
        years=50,
    )
    
    solver = solve_vao_eetc_boanbc(
        gcfg,
        train_instances,
        mip_gap=0.00,
        time_limit=7200,
        verbose=False,
        warm_start=True,
    )
    solver_sol = solver.get_solution()
    assert solver_sol is not None, "Solver did not find a solution." 
    assert solver_sol.vao_sol is not None, "VAO solution is None."
    assert solver_sol.SPs_sol is not None, "TC solutions are None."
    
    