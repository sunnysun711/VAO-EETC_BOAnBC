# solvers/BD_vao2_eetc.py
"""Benders Decomposition solver for VAO-M2 and EETC problems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from configs.ground import GroundConfigM2
from solvers.greedy_vao import generate_greedy_track
from utils.gurobi import (
    extract_gp_solution,
    set_warm_start_vars,
    set_params_for_robust_solve_socp,
    STATUS_NAMES
)
from utils.plotting import latexify, save_vao_plot, save_tc_plot

from models.vao2 import build_vao_model, generate_flat_track, get_vao_objective
from models.tc_socp import (
    build_active_constraints_lp,
    build_tc_fixed_model,
    build_tc_model,
    compute_optim_cut,
    update_tc_coupling_constraints,
    USE_ACTIVE_CONSTRAINTS_LP,
)
from solvers.base import Solution, SolverBase, TrainInstance, build_train_instances


latexify()


def save_iteration_solution(
    run_dir: Path,
    iteration: int,
    LB: float,
    UB: float,
    vao_sol: dict[str, dict[Any, float]],
    sp_records: dict[str, dict[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "iteration": iteration,
        "LB": float(LB) if LB not in [float("inf"), -float("inf")] else str(LB),
        "UB": float(UB) if UB not in [float("inf"), -float("inf")] else str(UB),
        "VAO": vao_sol,
        "SP": sp_records,
    }
    with open(run_dir / f"iter_{iteration:03d}/solution.json", "w") as f:
        json.dump(payload, f, indent=2)


def parse_benders_summary(json_file_path, solver_name, case_name) -> pd.DataFrame:
    """Parse Benders summary.json into a logtools-compatible DataFrame."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    start_time = datetime.fromisoformat(data['start_time'])
    df_data = []

    for iter_data in data['iterations']:
        iter_time = datetime.fromisoformat(iter_data['time'])
        elapsed_time = (iter_time - start_time).total_seconds()

        row = {
            'LogFilePath': os.path.abspath(json_file_path),
            'LogNumber': 1,
            'Incumbent': iter_data['UB'],
            'BestBd': iter_data['LB'],
            'Gap': iter_data['gap'],
            'Time': elapsed_time,
            'Seed': 0,
            'solver': solver_name,
            'case': case_name
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    return df


@dataclass
class BendersConfig:
    max_iter: int = 100
    mp_mip_gap: float = 0.0001
    mp_time_limit: float = 3600.0
    save_plots_every_iter: bool = True


class BendersSolver(SolverBase):
    def __init__(
        self,
        gcfg: GroundConfigM2,
        train_instances: list[TrainInstance],
        mip_gap: float = 0.01,
        time_limit: float = 3600.0,
        warm_start: bool = True,
        root_dir: str = "runs",
        name: str = "BD",
        verbose: bool = True,
        max_iter: int = 100,
        save_plots_every_iter: bool = True,
        is_soc_included: bool = True,
    ):
        super().__init__(
            gcfg, train_instances, time_limit, mip_gap, warm_start, root_dir, name, verbose
        )
        self.gcfg: GroundConfigM2 = gcfg
        self.train_instances: list[TrainInstance] = train_instances
        self.bcfg: BendersConfig = BendersConfig(
            max_iter=max_iter,
            mp_mip_gap=mip_gap,
            mp_time_limit=time_limit,
            save_plots_every_iter=save_plots_every_iter,
        )
        self.is_soc_included: bool = is_soc_included

        self.MP_model: Optional[gp.Model] = None
        self.MP_vars: dict[str, gp.tupledict] = {}
        self.MP_theta: gp.tupledict = gp.tupledict()
        self.MP_cs: dict[str, gp.tupledict] = {}
        self.MP_obj_exp: gp.LinExpr = gp.LinExpr()

        self.prev_vao_sol: dict[str, Any] = {}
        self.cur_vao_sol: dict[str, Any] = {}

        self.SPs: list[tuple[TrainInstance, gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]] = []

        self.LB = -float("inf")
        self.UB = float("inf")
        self.iteration: int = 0
        self.best_sol: Optional[dict[str, dict[Any, float]]] = None

    def build(self):
        self._build_master()
        self._set_warm_start()
        self._build_subproblems()

    def reset(self):
        self.LB = -float("inf")
        self.UB = float("inf")
        self.iteration: int = 0
        self.best_sol: Optional[dict[str, dict[Any, float]]] = None

    def _set_mp_log(self) -> None:
        assert self.MP_model is not None
        os.makedirs(self.run_dir / f"iter_{self.iteration:03d}", exist_ok=True)
        logfile = self.run_dir / f"iter_{self.iteration:03d}/MP_{self.iteration:03d}.log"
        self.MP_model.Params.LogFile = str(logfile)

    def _set_sp_log(self) -> None:
        assert len(self.SPs) > 0, "No subproblems to set log for."
        os.makedirs(self.run_dir / f"iter_{self.iteration:03d}", exist_ok=True)
        for ti, SP, _, _ in self.SPs:
            logfile = self.run_dir / f"iter_{self.iteration:03d}/SP_{self.iteration:03d}_{ti.name}.log"
            SP.Params.LogFile = str(logfile)

    def _build_master(self) -> None:
        MP_model, MP_vars, MP_cs = build_vao_model(
            self.gcfg, scale=self.gcfg.obj_scale,
            is_soc_included=self.is_soc_included,
        )
        MP_theta = gp.tupledict()
        for ti in self.train_instances:
            MP_theta[ti.name] = MP_model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_{ti.name}")
        MP_obj_exp: gp.LinExpr = get_vao_objective(MP_vars, S=self.gcfg.S, scale=self.gcfg.obj_scale)  # type: ignore
        MP_obj_exp += gp.quicksum(MP_theta[ti.name] for ti in self.train_instances)
        MP_model.setObjective(MP_obj_exp, gp.GRB.MINIMIZE)
        MP_model.update()

        MP_model.Params.LogFile = str(self.run_dir / "solve.log")
        MP_model.Params.LogToConsole = 0
        MP_model.Params.MIPGap = self.bcfg.mp_mip_gap
        MP_model.Params.TimeLimit = self.bcfg.mp_time_limit

        self.MP_model = MP_model
        self.MP_vars = MP_vars
        self.MP_theta = MP_theta
        self.MP_cs = MP_cs
        self.MP_obj_exp = MP_obj_exp

        self._set_mp_log()

    def _set_warm_start(self) -> None:
        try:
            self.cur_vao_sol = generate_greedy_track(self.gcfg)
        except Exception as e:
            self._log(f"Greedy track generation failed with error: {e}")
            self.cur_vao_sol = generate_flat_track(self.gcfg, is_mid_vpi=True)
        self.cur_vao_sol['theta'] = {ti.name: 0.0 for ti in self.train_instances}
        set_warm_start_vars(self.MP_vars, self.cur_vao_sol, debug_on=False, set_continuous=True)

    def _build_subproblems(self) -> None:
        assert self.cur_vao_sol is not {}, "No MP init solution provided."
        SPs = []
        for ti in self.train_instances:
            SP_model, SP_vars, SP_cs = build_tc_model(
                cfg=ti.train_config,
                S=self.gcfg.S,
                ds=self.gcfg.ds,
                is_uphill_dir=ti.is_uphill,
                vao_vars=self.cur_vao_sol,
                is_reg_included=False,
                obj_scale=ti.scale,
            )
            SP_model.Params.LogToConsole = 0
            set_params_for_robust_solve_socp(SP_model)

            SPs.append((ti, SP_model, SP_vars, SP_cs))

        self.SPs = SPs
        self._set_sp_log()

    def _solve_master(self):
        assert self.MP_model is not None, "No MP model built."
        self._set_mp_log()
        self.MP_model.optimize()
        status = self.MP_model.Status

        if self.verbose:
            self._log(
                f"  MP Status: {STATUS_NAMES.get(status, status)}, "
                f"obj={self.MP_model.ObjVal:.6e}, "
                f"gap={self.MP_model.MIPGap:.6e}",
            )

        if status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            if self.verbose:
                self._log(f" - MP not optimal, terminating.")
            return False

        self.LB = float(self.MP_model.ObjVal)
        self.prev_vao_sol = self.cur_vao_sol
        self.cur_vao_sol = extract_gp_solution(self.MP_vars)
        self.cur_vao_sol['theta'] = extract_gp_solution(self.MP_theta)
        if self.verbose:
            vao_obj_val = get_vao_objective(self.cur_vao_sol, S=self.gcfg.S, scale=self.gcfg.obj_scale)
            self._log(
                f"  MP VAO: ObjVal={vao_obj_val:.6e}, sum_theta={sum(self.cur_vao_sol['theta'].values()):.6e}"
            )

        if self.bcfg.save_plots_every_iter:
            save_vao_plot(
                run_dir=self.run_dir,
                file_prefix=f"iter_{self.iteration:03d}",
                file_suffix="",
                gcfg=self.gcfg,
                cur_sol=self.cur_vao_sol,
                prev_sol=self.prev_vao_sol,
            )
        return True

    def _solve_subproblems(self) -> tuple[bool, float, dict[str, dict[str, Any]]]:
        """Solve all SPs, add cuts, update UB."""
        assert self.MP_model is not None, "No MP model built."
        assert self.MP_vars is not None, "No MP vars provided."
        assert self.cur_vao_sol is not {}, "No MP init solution provided."

        sum_of_SP_obj = 0.0
        all_feasible = True
        sp_records: dict[str, dict[str, Any]] = {}
        self._set_sp_log()

        for ti, SP_m, SP_v, SP_c in self.SPs:
            update_tc_coupling_constraints(
                SP_m, SP_v, SP_c,   # type: ignore
                self.cur_vao_sol, self.gcfg,
                ti.train_config, ti.is_uphill,
            )
            SP_m.optimize()
            status = SP_m.Status

            rec: dict[str, Any] = {
                "name": ti.name,
                "status": int(status),
                "status_name": STATUS_NAMES.get(status, str(status)),
                "ObjVal": float("nan"),
                "vars": {},
            }

            if status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                all_feasible = False
                if self.verbose:
                    self._log(f"  SP {ti.name} Status: {STATUS_NAMES.get(status, status)}", )

            else:
                if self.bcfg.save_plots_every_iter:
                    save_tc_plot(
                        run_dir=self.run_dir,
                        file_prefix=f"iter_{self.iteration:03d}",
                        file_suffix="",
                        SP_vars=SP_v,
                        ti_name=ti.name,
                        ds=self.gcfg.ds,
                    )

                cut_rhs = compute_optim_cut(
                    SP_c, SP_m.ObjVal,
                    self.MP_vars, self.cur_vao_sol,
                    ti.train_config, self.gcfg,
                    is_wtn_included=True, is_uphill_dir=ti.is_uphill,
                )
                if cut_rhs is not None:
                    self.MP_model.addConstr(
                        self.MP_theta[ti.name] >= cut_rhs,
                        name=f"OptCut_{self.iteration}_{ti.name}",
                    )
                    self.MP_model.update()
                    rec["ObjVal"] = float(SP_m.ObjVal)
                    rec["vars"] = extract_gp_solution(SP_v)
                    sum_of_SP_obj += float(SP_m.ObjVal)

                    self._log(
                        f"  SP {ti.name}: "
                        f"current theta={self.cur_vao_sol['theta'][ti.name]:.4f}, "
                        f"TC_obj={float(SP_m.ObjVal):.4f}, added optim cut"
                    )
                else:
                    model_str = "active" if USE_ACTIVE_CONSTRAINTS_LP else "fixed"
                    self._log(f"  SP {ti.name}: DUAL EXTRACTION FAILED, try {model_str} LP now.")
                    socp_sol = extract_gp_solution(SP_v)
                    if USE_ACTIVE_CONSTRAINTS_LP:
                        fix_m, fix_v, fix_c, active_logs = build_active_constraints_lp(
                            ti.train_config, socp_sol,
                            S=self.gcfg.S, ds=self.gcfg.ds,
                            is_uphill_dir=ti.is_uphill, vao_vars=self.cur_vao_sol,
                            is_reg_included=False,
                            obj_scale=ti.scale,
                        )
                        for msg in active_logs:
                            self._log(f"    {msg}")
                    else:
                        fix_m, fix_v, fix_c = build_tc_fixed_model(
                            ti.train_config, socp_sol,
                            S=self.gcfg.S, ds=self.gcfg.ds,
                            is_uphill_dir=ti.is_uphill, vao_vars=self.cur_vao_sol,
                            is_reg_included=False,
                        )

                    fix_m.Params.LogToConsole = 0
                    fix_m.Params.LogFile = str(
                        self.run_dir / f"iter_{self.iteration:03d}/SP_{self.iteration:03d}_{ti.name}_{model_str}.log"
                    )
                    fix_m.optimize()

                    if fix_m.Status == GRB.OPTIMAL:
                        cut_rhs = compute_optim_cut(
                            fix_c, SP_m.ObjVal,
                            self.MP_vars, self.cur_vao_sol,
                            ti.train_config, self.gcfg,
                            is_wtn_included=True, is_uphill_dir=ti.is_uphill,
                        )
                        if cut_rhs is not None:
                            self.MP_model.addConstr(
                                self.MP_theta[ti.name] >= cut_rhs,
                                name=f"OptCut_{self.iteration}_{ti.name}",
                            )
                            self.MP_model.update()
                            rec["ObjVal"] = float(fix_m.ObjVal)
                            rec["vars"] = extract_gp_solution(fix_v)
                            rec["vars"]["v"] = socp_sol["v"]  # v is not in the fixed LP, add it back
                            sum_of_SP_obj += float(fix_m.ObjVal)
                            self._log(f"  SP {ti.name}: theta={self.cur_vao_sol['theta'][ti.name]:.4f}, "
                                    f"TC_obj={SP_m.ObjVal:.4f}, added optim cut ({model_str}-LP)")
                        else:
                            all_feasible = False
                            self._log(f"  SP {ti.name}: {model_str}-LP dual extraction failed, no optim cut added.")
                    else:
                        all_feasible = False
                        self._log(f"  SP {ti.name}: {model_str}-LP Status: {STATUS_NAMES.get(fix_m.Status, fix_m.Status)}")
                        if fix_m.Status == GRB.INFEASIBLE:
                            fix_m.computeIIS()
                            pat = self.run_dir / f"iter_{self.iteration:03d}/SP_{self.iteration:03d}_{ti.name}_{model_str}.ilp"
                            fix_m.write(str(pat))

            sp_records[ti.name] = rec

        if all_feasible:
            cur_vao_ObjVal: float = get_vao_objective(
                self.cur_vao_sol, self.gcfg.S, self.gcfg.obj_scale)  # type: ignore
            candidate_UB = cur_vao_ObjVal + sum_of_SP_obj
            if candidate_UB < self.UB:
                self.UB = candidate_UB
                self.best_sol = self.cur_vao_sol
                self.best_sol['SP'] = {}
                for idx, ti in enumerate(self.train_instances):
                    self.best_sol['SP'][idx] = sp_records[ti.name]["vars"]
                if self.verbose:
                    self._log(
                        f"  UB updated: {self.UB:.6e} "
                        f"(VAO={cur_vao_ObjVal:.6e} + TC={sum_of_SP_obj:.6e})"
                    )
        return all_feasible, sum_of_SP_obj, sp_records

    def solve(self):
        assert self.MP_model is not None
        assert self.cur_vao_sol is not None
        assert self.SPs is not None

        _START_TIME = datetime.now()

        if self.verbose:
            self._log("=" * 70)
            self._log(f"Benders Decomposition Started at {_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
            self._log(f"Run directory: {self.run_dir}")
            self._log(f"Number of segments: {self.gcfg.S}, Segment length: {self.gcfg.ds} m")
            self._log(f"Number of train instances: {len(self.train_instances)}")
            for ti in self.train_instances:
                self._log(f"  - {ti.name}: scale={ti.scale:.4f}, T={ti.train_config.T:.1f}s")
            self._log(f"Max iterations: {self.bcfg.max_iter}, MP MIP gap: {self.bcfg.mp_mip_gap:.2e}")
            self._log("=" * 70)

        summary: dict[str, Any] = {
            "run_dir": str(self.run_dir),
            "start_time": _START_TIME.isoformat(),
            "iterations": [],
        }

        for iteration in range(self.bcfg.max_iter):
            self.iteration = iteration

            if self.verbose:
                self._log(f"\n{'---' * 23}")
                self._log(f"Iteration {iteration:03d}")
                self._log(f"{'---' * 23}")

            iter_dir = self.run_dir / f"iter_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Solve master problem
            mp_success = self._solve_master()
            if not mp_success:
                break

            # Step 2: Solve all subproblems
            all_feasible, sum_of_SP_obj, sp_records = self._solve_subproblems()

            # Step 3: Save iteration results
            save_iteration_solution(
                run_dir=self.run_dir,
                iteration=iteration,
                LB=self.LB,
                UB=self.UB,
                vao_sol=self.cur_vao_sol,
                sp_records=sp_records,
            )

            # Step 4: Check convergence
            it_gap = None
            if self.UB < float("inf") and self.LB > -float("inf"):
                it_gap = (self.UB - self.LB) / max(abs(self.UB), 1e-10)
                if self.verbose:
                    self._log(f"  LB={self.LB:.6e}, UB={self.UB:.6e}, Gap={it_gap:.6%}")

            summary["iterations"].append(
                {
                    "iteration": iteration,
                    "time": datetime.now().isoformat(),
                    "LB": self.LB,
                    "UB": self.UB,
                    "gap": it_gap,
                    "all_feasible": all_feasible,
                }
            )

            if it_gap is not None and it_gap < self.bcfg.mp_mip_gap:
                if self.verbose:
                    self._log(f"\n  Converged! Gap {it_gap:.6%} < MP MIP gap {self.bcfg.mp_mip_gap:.6%}")
                break

        _END_TIME = datetime.now()

        if self.verbose:
            self._log("\n" + "=" * 70)
            self._log("Benders Decomposition Completed")
            self._log("=" * 70)
            self._log(f"Total iterations: {len(summary['iterations'])}")
            self._log(f"Solve time: {(_END_TIME - _START_TIME).total_seconds():.2f} seconds")
            self._log(f"Final LB: {self.LB:.6e}")
            self._log(f"Final UB: {self.UB:.6e}")
            if self.UB < float("inf") and self.LB > -float("inf"):
                final_gap = (self.UB - self.LB) / max(abs(self.UB), 1e-10)
                self._log(f"Final Gap: {final_gap:.4%}")
                if final_gap < 0:
                    self._log("WARNING: Negative gap detected! Check cut formulation.")
            self._log("=" * 70)

        summary["end_time"] = _END_TIME.isoformat()
        summary["final"] = {
            "LB": self.LB,
            "UB": self.UB,
            "gap": (
                (self.UB - self.LB) / max(abs(self.UB), 1e-10)
                if (self.UB < float("inf") and self.LB > -float("inf"))
                else None
            ),
            "solve_time": (_END_TIME - _START_TIME).total_seconds(),
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.log_f.close()

    def get_solution(self) -> Optional[Solution]:
        if self.best_sol is None:
            return None
        vao_sol = {k: v for k, v in self.best_sol.items() if k != "SP"}
        SPs_sol = self.best_sol["SP"]
        return Solution(vao_sol=vao_sol, SPs_sol=SPs_sol)  # type: ignore


def main() -> None:
    gcfg = GroundConfigM2(ground_name="HP3")

    train_instances = build_train_instances(
        gcfg, {},
        ti_info=[("NL_intercity_VIRM6_", (True, False), (120, 120), (1093, 1088))],
        safety_margin=1.15,
        years=50,
    )

    solver = BendersSolver(
        gcfg=gcfg,
        train_instances=train_instances,
        max_iter=100,
        mip_gap=0.00,
        time_limit=600,
        save_plots_every_iter=True,
        root_dir="runs",
        is_soc_included=True,
        verbose=True
    )
    solver.build()
    solver.solve()
    solver.save_solution()

    assert solver.best_sol is not None


if __name__ == "__main__":
    main()