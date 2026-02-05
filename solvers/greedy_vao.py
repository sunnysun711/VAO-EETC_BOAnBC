# solvers/greedy_vao.py

import random
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from utils.gurobi import STATUS_NAMES, extract_gp_solution
from configs.ground import GroundConfigBase
from models.vao2 import rg_all, rg_curv, rg_grad, rg_plat, generate_flat_track

def generate_greedy_track(cfg: GroundConfigBase) -> dict[str, Any]:
    sigma = cfg.sigma
    hbar = dict(enumerate(cfg.e6g, start=1))  # start from 1, same index system as variable `e`
    S, ds = cfg.S, cfg.ds
    emi, ema = cfg.y_range
    S_plat = cfg.platform_intervals
    
    m = gp.Model("flat_track")
    e  = m.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=emi, ub=ema, name="e")   # real ele
    pi = m.addVars(rg_all(S), vtype=GRB.BINARY, name="pi")  # VPI points
    variables = { "e": e, "pi": pi, }
    
    m.addConstrs((gp.quicksum(pi[s] for s in range(i, i+sigma)) <= 1 for i in range(1, S - sigma + 2)), name="VPISpacing", )
    m.addConstrs((-cfg.di_max * ds * pi[s] - e[s+1] - e[s-1] + 2*e[s] <= 0 for s in rg_curv(S)), name="CurvatureLB", )
    m.addConstrs((e[s + 1] + e[s - 1] - 2 * e[s] - cfg.di_max * ds * pi[s] <= 0 for s in rg_curv(S)), name="CurvatureUB", )
    m.addConstrs((e[s] - e[s+1] - cfg.i_max * ds <= 0 for s in rg_grad(S)), name="GradientLB", )
    m.addConstrs((e[s+1] - e[s] - cfg.i_max * ds <= 0 for s in rg_grad(S)), name="GradientUB", )
    for i in rg_plat(S, S_plat):
        m.addConstr(e[i] == hbar[i], name=f"Platform[{i}]")
    
    m.setObjective(gp.quicksum((e[s] - hbar[s])**2 for s in rg_all(S)), gp.GRB.MINIMIZE)
    
    for s in range(S_plat+1, S + 1 - S_plat - sigma + 1):
        if (s-S_plat-1) % sigma == 0:
            m.addConstr(pi[s] == 1)
        else:
            m.addConstr(pi[s] == 0)
    m.addConstr(pi[S+1-S_plat] == 1)
    
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = 10.0
    m.Params.MIPGap = 0.01
    m.optimize()
    # if m.Status == GRB.INFEASIBLE:
    #     m.computeIIS()
    #     m.write("iis.ilp")
        # m.write('full.lp')
    assert m.SolCount > 0, f"Greedy track generation failed with status {STATUS_NAMES.get(m.Status, m.Status)}"
    sol = extract_gp_solution(variables)
    
    sol['z1'], sol['z2'], sol['z3'] = {}, {}, {}
    sol['et'], sol['e0'], sol['eb'] = {}, {}, {}
    for s in rg_all(S):
        grd = hbar[s]
        ele = sol['e'][s]
        
        sol['et'][s] = 0.0
        sol['e0'][s] = 0.0
        sol['eb'][s] = 0.0
        sol['z1'][s] = 0.0
        sol['z2'][s] = 0.0
        sol['z3'][s] = 0.0
        
        if ele <= grd - cfg.ht_min:
            sol['et'][s] = ele
            sol['z1'][s] = 1.0
        elif ele >= grd + cfg.hb_min:
            sol['eb'][s] = ele
            sol['z3'][s] = 1.0
        else:
            sol['e0'][s] = ele
            sol['z2'][s] = 1.0
    return sol


if __name__ == "__main__":
    from configs.ground import GroundConfigM2
    gcfg = GroundConfigM2(
        num_eval_points=801, ds=50, smooth_scale=0.8,
    )
    
    sol = generate_greedy_track(gcfg)
    # sol = generate_flat_track(gcfg)
    print(f"sigma = {gcfg.sigma} | S_plat = {gcfg.platform_intervals} | S = {gcfg.S} | Platforms: {list(rg_plat(gcfg.S, gcfg.platform_intervals))}")
    print(f"VPI points: {[k for k, v in sol['pi'].items() if v == 1]}")
    
    import matplotlib.pyplot as plt
    from utils.plotting import draw_vao
    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    draw_vao(
        ax, gcfg, sol, show_ground=True, show_track=True, show_tunnel=True, show_bridge=True, 
        style_track=dict(ls=":", alpha=0.8, lw=0.7, label="track"), 
        style_vpi=dict(marker="x", color="gray", label="VPI points")
    )
    fig.tight_layout()
    plt.show()
    pass