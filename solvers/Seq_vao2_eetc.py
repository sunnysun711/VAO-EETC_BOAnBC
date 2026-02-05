# solvers/Seq_vao2_eetc.py
"""Sequential two-step model for VAO2 and EETC."""

from typing import Optional
import gurobipy as gp

from utils.gurobi import STATUS_NAMES, set_warm_start_vars, extract_gp_solution, set_params_for_robust_solve_socp
from configs.ground import GroundConfigM2
from models.vao2 import build_vao_model, generate_flat_track, get_vao_objective
from models.tc_socp import build_tc_model
from solvers.base import Solution, SolverBase, TrainInstance, build_train_instances
from solvers.greedy_vao import generate_greedy_track

class SeqSolver(SolverBase):
    def __init__(
        self, 
        gcfg: GroundConfigM2, 
        train_instances: list[TrainInstance],
        mip_gap: float = 0.01,
        time_limit: float = 600.0,
        warm_start: bool = True,
        root_dir: str = "runs",
        name: str = "Seq",
        verbose: bool = True,
    ):
        super().__init__(
            gcfg, train_instances, time_limit, mip_gap, warm_start, root_dir, name, verbose
        )
        self.gcfg: GroundConfigM2 = gcfg
        self.train_instances = train_instances
        
        self.vao_m: Optional[gp.Model] = None
        self.vao_v: Optional[dict] = None
        self.vao_c: Optional[dict] = None
        
        self.SPs: dict[int, gp.Model] = {}
        
    def build(self):
        """Build master model (VAO)"""
        gcfg = self.gcfg
        
        vao_m, vao_v, vao_c = build_vao_model(
            gcfg, scale=gcfg.obj_scale, is_soc_included=True
        )
        hbar = dict(enumerate(gcfg.e6g, start=1))
        vao_m.setObjective(
            gp.quicksum((vao_v["e"][s] - hbar[s]) ** 2 for s in range(1, gcfg.S + 2)),
            gp.GRB.MINIMIZE,
        )
        
        vao_m.Params.LogFile = str(self.run_dir / f"solve.log")
        vao_m.Params.OutputFlag = 1
        vao_m.Params.TimeLimit = self.time_limit
        vao_m.Params.MIPGap = self.mip_gap
        vao_m.Params.NumericFocus = 3
        vao_m.Params.FeasibilityTol = 1e-6
        if self.warm_start:
            try:
                ws_sol = generate_greedy_track(gcfg)
                self._log("Using greedy track as initial solution")
            except Exception as e:
                self._log(f"Greedy track failed ({e}), using flat track")
                ws_sol = generate_flat_track(gcfg, is_mid_vpi=True)
            set_warm_start_vars(vao_v, ws_sol, debug_on=False, set_continuous=True)
        
        obj_expr = get_vao_objective(vao_v, gcfg.S, scale=gcfg.obj_scale)
        vao_m.setObjective(obj_expr, gp.GRB.MINIMIZE)
        
        self.vao_m = vao_m
        self.vao_v = vao_v
        self.vao_c = vao_c
        return
        
    def solve(self):
        gcfg = self.gcfg
        train_instances = self.train_instances
        vao_m = self.vao_m
        vao_v = self.vao_v
        assert vao_m is not None, "VAO model not built"
        assert vao_v is not None, "VAO variables not built"
        
        vao_m.optimize()
        assert vao_m.SolCount > 0, (
            f"[Warning] VAO optimization failed: "
            f"{STATUS_NAMES.get(vao_m.Status, vao_m.Status)}"
        )
        vao_sol = extract_gp_solution(vao_v)
        vao_obj = vao_m.ObjVal
        self._log(f"VAO optimization completed: obj={vao_obj:.4f}")

        for idx, ti in enumerate(train_instances):
            self._log(f"Solving EETC-{idx:02d} ({ti.name})...")
            tc_m, tc_v, tc_c = build_tc_model(
                ti.train_config, S=gcfg.S, ds=gcfg.ds, 
                is_uphill_dir=ti.is_uphill, 
                vao_vars=vao_sol, 
                is_reg_included=False, 
                obj_scale=ti.scale
            )
            tc_m._variables = tc_v
            tc_m._constraints = tc_c
            tc_m.Params.OutputFlag = 0
            set_params_for_robust_solve_socp(tc_m)
            tc_m.optimize()
            self.SPs[idx] = tc_m
            
            assert tc_m.SolCount > 0, (
                f"[Warning] TC optimization failed: "
                f"{STATUS_NAMES.get(tc_m.Status, tc_m.Status)}"
            )
            self._log(
                f"Status: {STATUS_NAMES.get(tc_m.Status, tc_m.Status)}, "
                f"TC_obj: {tc_m.ObjVal:.4f}"
            )
    
    def get_solution(self) -> Solution:
        """Get solution of the solver.
        
        Returns:
            vao_sol: Solution of the VAO model
            SPs_sol: Solutions of the TC subproblems
        """
        assert self.vao_v is not None, "VAO variables not built"
        assert self.SPs, "TC subproblems not solved"
        vao_sol = extract_gp_solution(self.vao_v)
        SPs_sol = {idx: extract_gp_solution(tc_m._variables) for idx, tc_m in self.SPs.items()}
        sol = Solution(vao_sol, SPs_sol)
        return sol
    
if __name__ == "__main__":
    # gcfg = GroundConfigM2(
    #     num_eval_points=120, 
    #     ds=50, 
    #     # smooth_scale=0.5,
    #     # seed=1902132
    # )
    gcfg = GroundConfigM2("HP2")
    ws_sol = generate_greedy_track(gcfg)
    train_instances = build_train_instances(
        gcfg, ws_sol, 
        ti_info=[("CRH380AL_", (True, False), (180, 200), (None, None)), ("HXD1D_", (False, True), (30, 20), (None, None))],
        safety_margin=1.2, 
        years=50
    )
    
    solver = SeqSolver(gcfg, train_instances, mip_gap=0.00, time_limit=600.0, warm_start=True)
    solver.build()
    solver.solve()
    solver.save_solution()
    pass