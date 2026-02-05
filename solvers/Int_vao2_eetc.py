# solvers/Int_vao2_eetc.py
"""Integrated model for VAO2 and EETC."""

from typing import Optional
import gurobipy as gp

from utils.gurobi import (
    STATUS_NAMES,
    set_warm_start_vars,
    extract_gp_solution,
    set_params_for_robust_solve_socp,
)
from configs.ground import GroundConfigM2
from models.vao2 import build_vao_model
from models.tc_socp import (
    add_tc_variables,
    build_tc_model,
    add_eetc_objective,
    add_tc_constraints,
)
from solvers.base import SolverBase, Solution, TrainInstance, build_train_instances
from solvers.greedy_vao import generate_greedy_track


class IntSolver(SolverBase):
    """Integrated solver for VAO2 and EETC optimization."""

    USE_MODEL_EQ_CONSTR = True

    # Gurobi parameter presets
    WARM_START_PARAMS = {
        "OutputFlag": 0,
        "SubMIPNodes": 500,
        "SolutionLimit": 1,
        "TimeLimit": 30.0,
        "FeasibilityTol": 1e-5,
        "NumericFocus": 3,
        "ScaleFlag": 2,
    }

    SOLVE_PARAMS = {
        "OutputFlag": 1,
        "SubMIPNodes": 50000,
        "FeasibilityTol": 1e-4,
        "NumericFocus": 3,
        "ScaleFlag": 2,
    }

    def __init__(
        self,
        gcfg: GroundConfigM2,
        train_instances: list[TrainInstance],
        mip_gap: float = 0.01,
        time_limit: float = 600.0,
        warm_start: bool = True,
        root_dir: str = "runs",
        name: str = "Int",
        verbose: bool = True,
    ):
        super().__init__(
            gcfg, train_instances, time_limit, mip_gap, warm_start, root_dir, name, verbose
        )
        self.gcfg: GroundConfigM2 = gcfg
        self.train_instances = train_instances

        self.model: Optional[gp.Model] = None
        self.variables: Optional[dict] = None
        self.constraints: Optional[dict] = None
        self.tc_var_dict: Optional[dict] = None
        self.tc_cs_dict: Optional[dict] = None

    def build(self):
        int_model, int_vars, int_cs = build_vao_model(
            self.gcfg, scale=self.gcfg.obj_scale, is_soc_included=True
        )

        tc_var_dict, tc_cs_dict = self._add_train_constraints(int_model, int_vars)

        if self.warm_start:
            self._apply_warm_start(int_model, int_vars, tc_var_dict)

        self._configure_solve_params(int_model)

        self.model = int_model
        self.variables = int_vars
        self.constraints = int_cs
        self.tc_var_dict = tc_var_dict
        self.tc_cs_dict = tc_cs_dict

    def _add_train_constraints(
        self, model: gp.Model, vao_vars: dict
    ) -> tuple[dict, dict]:
        tc_var_dict = {}
        tc_cs_dict = {}

        for idx, ti in enumerate(self.train_instances):
            tc_var_dict[idx] = add_tc_variables(
                model, ti.train_config, S=self.gcfg.S
            )
            tc_cs_dict[idx] = add_tc_constraints(
                model,
                ti.train_config,
                vao_vars=vao_vars,
                variables=tc_var_dict[idx],
                S=self.gcfg.S,
                ds=self.gcfg.ds,
                is_uphill_dir=ti.is_uphill,
                cs_name_prefix=f"T{idx}_",
            )
            add_eetc_objective(
                model,
                ti.train_config,
                variables=tc_var_dict[idx],
                S=self.gcfg.S,
                ds=self.gcfg.ds,
                scale=ti.scale,
                is_reg_included=False,
            )

        return tc_var_dict, tc_cs_dict

    def _apply_warm_start(
        self, model: gp.Model, vao_vars: dict, tc_var_dict: dict
    ):
        self._log("Generating warm start...")
        ws_sol = self._get_warm_start()

        self._log("Setting warm start solution...")
        
        self._set_warm_start_constraints(
            model, vao_vars, tc_var_dict, ws_sol,
            debug_on=False,
        )

    def _set_warm_start_constraints(
        self,
        model: gp.Model,
        vao_vars: dict,
        tc_var_dict: dict,
        ws_sol: Solution,
        debug_on: bool = True,
    ) -> dict:
        ws_kwargs = {
            "debug_on": debug_on,
            "set_continuous": True,
            "use_model_eq_constr": self.USE_MODEL_EQ_CONSTR,
            "model": model,
            "suppress_warning": True,
        }
        assert ws_sol.SPs_sol is not None, "Warm start solution for TC subproblems is None."
        vao_ws_constr = set_warm_start_vars(vao_vars, ws_sol.vao_sol, **ws_kwargs)

        tc_ws_constrs = []
        for idx in range(len(self.train_instances)):
            tc_constr = set_warm_start_vars(
                tc_var_dict[idx], ws_sol.SPs_sol[idx], **ws_kwargs
            )
            tc_ws_constrs.append(tc_constr)

        return {"vao": vao_ws_constr, "tc": tc_ws_constrs}

    def _run_warm_start_optimization(self, model: gp.Model):
        """Run initial optimization to establish warm start."""
        for param, value in self.WARM_START_PARAMS.items():
            setattr(model.Params, param, value)

        model.optimize()

        if model.SolCount == 0:
            self._log(f"[WARN] Warm start failed with status: {STATUS_NAMES[model.Status]}")
            # if model.Status == GRB.INFEASIBLE:
            #     model.computeIIS()
            #     model.write("ws_infeasible.ilp")

    def _remove_warm_start_constraints(self, model: gp.Model, ws_constrs: dict):
        """Remove temporary warm start equality constraints."""
        if not self.USE_MODEL_EQ_CONSTR:
            return

        for constr_dict in ws_constrs["vao"].values():
            model.remove(constr_dict)

        for tc_ws_constr in ws_constrs["tc"]:
            for constr_dict in tc_ws_constr.values():
                model.remove(constr_dict)

    def _configure_solve_params(self, model: gp.Model):
        """Configure Gurobi parameters for the main solve."""
        model.resetParams()

        model.Params.LogFile = str(self.run_dir / "solve.log")
        for param, value in self.SOLVE_PARAMS.items():
            setattr(model.Params, param, value)

        model.Params.MIPGap = self.mip_gap
        model.Params.TimeLimit = self.time_limit

    def solve(self):
        assert self.model is not None, "Model not built yet"
        self.model.optimize()
        return

    def get_solution(self) -> Optional[Solution]:
        assert self.model is not None, "Model not built yet"
        assert self.variables is not None, "Variables not built yet"
        assert self.tc_var_dict is not None, "TC variables not built yet"

        if self.model.SolCount == 0:
            return None

        vao_sol = extract_gp_solution(self.variables)
        tc_sol_dict = {
            idx: extract_gp_solution(self.tc_var_dict[idx])
            for idx in range(len(self.train_instances))
        }
        return Solution(vao_sol=vao_sol, SPs_sol=tc_sol_dict)


    def _get_warm_start(self):
        vao_sol = generate_greedy_track(self.gcfg)
        tc_vars_dict = {}
        for idx, ti in enumerate(self.train_instances):
            self._log(f"Solving EETC-{idx:02d} ({ti.name})...")

            tc_m, tc_v, tc_c = build_tc_model(
                ti.train_config,
                S=self.gcfg.S,
                ds=self.gcfg.ds,
                is_uphill_dir=ti.is_uphill,
                vao_vars=vao_sol,
                is_reg_included=False,
                obj_scale=ti.scale,
                tighten_time_limit=0.1,  # ensure feasibility in integrated model
            )

            tc_m.Params.OutputFlag = 0
            set_params_for_robust_solve_socp(tc_m)
            tc_m.optimize()

            assert tc_m.SolCount > 0, (
                f"TC optimization failed: {STATUS_NAMES.get(tc_m.Status, tc_m.Status)}"
            )
            self._log(f"Status: {STATUS_NAMES.get(tc_m.Status, tc_m.Status)}, TC_obj: {tc_m.ObjVal:.4f}")

            tc_vars_dict[idx] = extract_gp_solution(tc_v)
        
        return Solution(vao_sol, tc_vars_dict)


if __name__ == "__main__":
    gcfg = GroundConfigM2(
        num_eval_points=50, ds=50,
        # smooth_scale=0.5,
        # seed=1902132,
    )
    # gcfg = GroundConfigM2("LD10")
    # ws_sol = generate_flat_track(gcfg, is_mid_vpi=True)
    ws_sol = generate_greedy_track(gcfg)

    train_instances = build_train_instances(
        gcfg,
        ws_sol,
        ti_info=[
            ("CRH380AL_", (True, False), (100, 120), (None, None)),
            ("HXD1D_", (False, True), (30, 20), (None, None)),
        ],
        safety_margin=1.3,
        years=50,
    )

    solver = IntSolver(
        gcfg, train_instances, mip_gap=0.02, time_limit=180.0, warm_start=True
    )
    solver.build()
    solver.solve()
    solver.save_solution()
    pass