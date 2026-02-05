# solvers/SeqBnOA_vao2_eetc.py
"""Sequential two-step model for VAO2 and EETC with BnOA structure.
    - Step 1: Solve VAO2 model with BnOA structure.
    - Step 2: Solve EETC model for each train instance.
"""

from typing import Optional
import gurobipy as gp

from utils.gurobi import STATUS_NAMES, set_warm_start_vars, extract_gp_solution, set_params_for_robust_solve_socp
from configs.ground import GroundConfigM2
from models.vao2 import build_vao_oa_master_model, generate_flat_track, get_vao_objective
from models.tc_socp import build_tc_model
from solvers.base import Solution, SolverBase, TrainInstance, build_train_instances
from solvers.greedy_vao import generate_greedy_track
from solvers.BnOA_vao2 import OA_CUT_MIN_H, OASolverStats, oa_callback

class SeqBnOASolver(SolverBase):
    """Sequential two-step model for VAO2 and EETC with BnOA structure.
    """
    def __init__(
        self, 
        gcfg: GroundConfigM2, 
        train_instances: list[TrainInstance],
        mip_gap: float = 0.01,
        time_limit: float = 600.0,
        warm_start: bool = True,
        root_dir: str = "runs",
        name: str = "SeqBnOA",
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
        
        self.stats = OASolverStats()
        
        self.SPs: dict[int, gp.Model] = {}
        
    def build(self):
        gcfg = self.gcfg
        
        vao_m, vao_v, vao_c = build_vao_oa_master_model(gcfg)
        self.vao_m = vao_m
        self.vao_v = vao_v
        self.vao_c = vao_c
        
        self._add_initial_cuts()
        self._log(f"Master problem built: {vao_m.NumVars} vars, "
                 f"{vao_m.NumConstrs} constrs")
        
        vao_m.Params.LogFile = str(self.run_dir / f"solve.log")
        vao_m.Params.OutputFlag = 1
        vao_m.Params.LazyConstraints = 1
        vao_m.Params.TimeLimit = self.time_limit
        vao_m.Params.MIPGap = self.mip_gap
        vao_m.Params.NumericFocus = 2
        vao_m.Params.FeasibilityTol = 1e-4
        
        if self.warm_start:
            self._log("Warm start enabled, generating initial solution...")
            try:
                ws_sol = generate_greedy_track(gcfg)
                self._log("Using greedy track as initial solution")
            except Exception as e:
                self._log(f"Greedy track failed ({e}), using flat track")
                ws_sol = generate_flat_track(gcfg, is_mid_vpi=True)
            set_warm_start_vars(vao_v, ws_sol, debug_on=False, set_continuous=True)
        
        obj_expr = get_vao_objective(vao_v, gcfg.S, scale=gcfg.obj_scale)
        vao_m.setObjective(obj_expr, gp.GRB.MINIMIZE)
        return

    def _add_initial_cuts(self):
        assert self.vao_m is not None, "Build master problem first"
        
        S = self.gcfg.S
        n_initial_cuts = 0
        
        Hb = self.gcfg.hb_min
        Ht = self.gcfg.ht_min
        H_max = self.gcfg.y_range[1] - self.gcfg.y_range[0]
        
        hf = self.vao_v["hf"]    # type: ignore
        hc = self.vao_v["hc"]    # type: ignore
        hb = self.vao_v["hb"]    # type: ignore
        hf2 = self.vao_v["hf2"]  # type: ignore
        hc2 = self.vao_v["hc2"]  # type: ignore
        hb2 = self.vao_v["hb2"]  # type: ignore
        
        for s in range(1, S + 2):
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
        gcfg = self.gcfg
        train_instances = self.train_instances
        vao_m = self.vao_m
        vao_v = self.vao_v
        assert vao_m is not None, "VAO model not built"
        assert vao_v is not None, "VAO variables not built"
        
        self.stats.reset()
        vao_m._cfg = self.gcfg
        vao_m._all_vars = self.vao_v
        vao_m._solver = self
        
        self._log("=" * 60)
        self._log("Starting Branch-and-Outer-Approximation optimization...")
        self._log("=" * 60)
        
        vao_m.optimize(oa_callback)
        
        self._log("=" * 60)
        self._log(f"Optimization finished in {self.stats.solve_time:.2f} seconds")
        self._log("=" * 60)
        
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
    
    def report(self):
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
    
    def get_solution(self) -> Solution:
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
        ti_info=[("NL_intercity_VIRM6_", (True, False), (120, 120), (855, 850))],
        safety_margin=1.15, 
        years=50
    )
    
    solver = SeqBnOASolver(gcfg, train_instances, mip_gap=0.00, time_limit=600.0, warm_start=True, verbose=False)
    solver.build()
    solver.solve()
    solver.report()
    solver.save_solution()
    pass