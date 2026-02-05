"""Vertical Alignment Optimization Model - M2 (Piecewise quadratic cost functions)."""
from dataclasses import dataclass
from typing import Any
import gurobipy as gp
from gurobipy import GRB
from configs import GroundConfigM2
from utils.gurobi import STATUS_NAMES, extract_dual_info


COST_SCALE = 1000.0

rg_all = lambda S: range(1, S+2)
rg_plat = lambda S, S_plat: list(range(1, S_plat+1)) + list(range(S+1-S_plat, S+2))
rg_curv = lambda S: range(2, S+1)
rg_grad = lambda S: range(1, S+1)


def add_mp_variables(model: gp.Model, cfg: GroundConfigM2) -> dict[str, gp.tupledict]:
    """Add VAO-M2-MP variables."""
    S, ds = cfg.S, cfg.ds
    c_tn, c_bg, c6e_tn = cfg.c_tn, cfg.c_bg, cfg.c6e_tn
    
    pi = model.addVars(rg_all(S), vtype=GRB.BINARY, name="pi")
    z1 = model.addVars(rg_all(S), vtype=GRB.BINARY, name="z1")
    z2 = model.addVars(rg_all(S), vtype=GRB.BINARY, name="z2")
    z3 = model.addVars(rg_all(S), vtype=GRB.BINARY, name="z3")
    
    Ct = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=c_tn * ds / COST_SCALE, name="Ct")
    Cb = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=c_bg * ds / COST_SCALE, name="Cb")
    Ctne = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=c6e_tn * ds / COST_SCALE, name="Ctne")

    model.update()
    
    return {
        "z1": z1, "z2": z2, "z3": z3, "pi": pi,
        "Ct": Ct, "Cb": Cb, "Ctne": Ctne,
    }


def add_mp_constraints(model: gp.Model, cfg: GroundConfigM2, mp_vars: dict[str, gp.tupledict]) -> dict[str, gp.tupledict]:
    """Add VAO-M2-MP constraints."""
    S, ds = cfg.S, cfg.ds
    sigma = cfg.sigma
    c_tn, c_bg, c6e_tn = cfg.c_tn, cfg.c_bg, cfg.c6e_tn
    
    z1, z2, z3, pi = mp_vars["z1"], mp_vars["z2"], mp_vars["z3"], mp_vars["pi"]
    Ct, Cb, Ctne = mp_vars["Ct"], mp_vars["Cb"], mp_vars["Ctne"]
    
    constrs = {}
    
    constrs["ModeSum"] = model.addConstrs((z1[s] + z2[s] + z3[s] == 1 for s in rg_all(S)), name="ModeSum")
    constrs["VPISpacing"] = model.addConstrs((gp.quicksum(pi[s] for s in range(i, i+sigma)) <= 1 for i in range(1, S - sigma + 2)), name="VPISpacing")
    model.addConstr(z1[S + 1] == 0, name="BoundaryZ1")
    constrs["CostTunnel"] = model.addConstrs((c_tn * ds * z1[s] / COST_SCALE - Ct[s] == 0 for s in rg_all(S)), name="CostTunnel")
    constrs["CostBridge"] = model.addConstrs((c_bg * ds * z3[s] / COST_SCALE - Cb[s] == 0 for s in rg_all(S)), name="CostBridge")
    constrs["TunnelEntryA"] = model.addConstrs((-Ctne[s] + c6e_tn * (z1[s] - z1[s+1]) / COST_SCALE <= 0 for s in range(1, S+1)), name="TunnelEntryA")
    constrs["TunnelEntryB"] = model.addConstrs((-Ctne[s] + c6e_tn * (z1[s+1] - z1[s]) / COST_SCALE <= 0 for s in range(1, S+1)), name="TunnelEntryB")
    
    model.update()
    return constrs


def add_sp_variables(model: gp.Model, cfg: GroundConfigM2) -> dict[str, gp.tupledict]:
    """Add VAO-M2-SP variables."""
    S, ds = cfg.S, cfg.ds
    Ht, Hb = cfg.ht_min, cfg.hb_min
    Me, We = cfg.Me, cfg.We
    U_surface = cfg.U_surface
    U_cut, U_fill = cfg.U_cut, cfg.U_fill
    c_pile, L_span = cfg.c_pl, cfg.L_span
    r_b = cfg.r_b
    emi, ema = cfg.y_range

    e = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=emi, ub=ema, name="e")
    e0 = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=ema, name="e0")
    et = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=ema-Ht, name="et")
    eb = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=ema, name="eb")

    earth_cost_max_soc = (max((We*Ht + Me * Ht**2)*U_cut, (We*Hb + Me * Hb**2)*U_fill) + U_surface) * ds
    earth_cost_max_linear = (max((We + Me) * Ht * U_cut, (We + Me) * Hb * U_fill) + U_surface) * ds
    earth_cost_max = max(earth_cost_max_soc, earth_cost_max_linear)
    Ce = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=earth_cost_max / COST_SCALE, name="Ce")
    
    H_max = ema - emi
    pile_cost_max_soc = c_pile * (H_max + r_b * H_max ** 2) * ds / L_span
    pile_cost_max_linear = c_pile * 2 * H_max * ds / L_span
    pile_cost_max = max(pile_cost_max_soc, pile_cost_max_linear)
    Cp = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=pile_cost_max / COST_SCALE, name="Cp")
    
    hf = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=Hb, name="hf")
    hc = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=Ht, name="hc")
    hb = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=ema-emi, name="hb")
    hf2 = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=Hb**2, name="hf2")
    hc2 = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=Ht**2, name="hc2")
    hb2 = model.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=(ema-emi)**2, name="hb2")

    model.update()

    return {
        "e": e, "e0": e0, "et": et, "eb": eb,
        "Ce": Ce, "Cp": Cp,
        "hf": hf, "hc": hc, "hb": hb,
        "hf2": hf2, "hc2": hc2, "hb2": hb2,
    }


def add_sp_constraints(
    mo: gp.Model, 
    cfg: GroundConfigM2, 
    sp_vars: dict[str, gp.tupledict],
    mp_vars: dict[str, gp.tupledict] | dict[str, dict[int, float]],
    is_height_included: bool = True,
    is_soc_included: bool = True,
    is_q_cost_included: bool = True,
    is_l_cost_included: bool = False,
) -> dict[str, gp.tupledict]:
    """Add VAO-M2-SP constraints."""
    assert not (is_q_cost_included and is_l_cost_included), "At most one of is_q_cost_included or is_l_cost_included can be True."
    
    S, ds = cfg.S, cfg.ds
    S_plat = cfg.platform_intervals
    hbar = dict(enumerate(cfg.e6g, start=1))
    i_max, di = cfg.i_max, cfg.di_max
    Ht, Hb = cfg.ht_min, cfg.hb_min
    Me, We = cfg.Me, cfg.We
    U_surface = cfg.U_surface
    U_cut, U_fill = cfg.U_cut, cfg.U_fill
    c_pl, L_span = cfg.c_pl, cfg.L_span
    r_b = cfg.r_b
    emi, ema = cfg.y_range
        
    e, et, e0, eb = sp_vars["e"], sp_vars["et"], sp_vars["e0"], sp_vars["eb"]
    Ce, Cp = sp_vars["Ce"], sp_vars["Cp"]
    hf, hc, hb = sp_vars["hf"], sp_vars["hc"], sp_vars["hb"]
    hf2, hc2, hb2 = sp_vars["hf2"], sp_vars["hc2"], sp_vars["hb2"]
    
    z1, z2, z3, pi = mp_vars['z1'], mp_vars['z2'], mp_vars['z3'], mp_vars['pi']
    
    constrs = {}
    
    constrs["ElevDecomp"] = mo.addConstrs((e[s] - et[s] - e0[s] - eb[s] == 0 for s in rg_all(S)), name="ElevDecomp")
    constrs["ElevTunnelLB"] = mo.addConstrs((emi * z1[s] - et[s] <= 0 for s in rg_all(S)), name="ElevTunnelLB")
    constrs["ElevTunnelUB"] = mo.addConstrs((et[s] - (hbar[s] - Ht) * z1[s] <= 0 for s in rg_all(S)), name="ElevTunnelUB")
    constrs["ElevEarthLB"] = mo.addConstrs(((hbar[s] - Ht) * z2[s] - e0[s] <= 0 for s in rg_all(S)), name="ElevEarthLB")
    constrs["ElevEarthUB"] = mo.addConstrs((e0[s] - (hbar[s] + Hb) * z2[s] <= 0 for s in rg_all(S)), name="ElevEarthUB")
    constrs["ElevBridgeLB"] = mo.addConstrs(((hbar[s] + Hb) * z3[s] - eb[s] <= 0 for s in rg_all(S)), name="ElevBridgeLB")
    constrs["ElevBridgeUB"] = mo.addConstrs((eb[s] - ema * z3[s] <= 0 for s in rg_all(S)), name="ElevBridgeUB")
    constrs["CurvatureLB"] = mo.addConstrs((-di * ds * pi[s] - e[s+1] - e[s-1] + 2*e[s] <= 0 for s in rg_curv(S)), name="CurvatureLB")
    constrs["CurvatureUB"] = mo.addConstrs((e[s + 1] + e[s - 1] - 2 * e[s] - di * ds * pi[s] <= 0 for s in rg_curv(S)), name="CurvatureUB")
    constrs["GradientLB"] = mo.addConstrs((e[s] - e[s+1] - i_max * ds <= 0 for s in rg_grad(S)), name="GradientLB")
    constrs["GradientUB"] = mo.addConstrs((e[s+1] - e[s] - i_max * ds <= 0 for s in rg_grad(S)), name="GradientUB")
    
    constrs["Platform"] = gp.tupledict()
    for i in rg_plat(S, S_plat):
        constrs["Platform"][i] = mo.addConstr(e[i] == hbar[i], name=f"Platform[{i}]")
    
    if is_height_included:
        constrs["HeightEarth"] = mo.addConstrs((e0[s] - hf[s] + hc[s] == hbar[s] * z2[s] for s in rg_all(S)), name="HeightEarth")
        constrs["HeightBridge"] = mo.addConstrs((eb[s] - hb[s] == hbar[s] * z3[s] for s in rg_all(S)), name="HeightBridge")
        constrs["BndFill"] = mo.addConstrs((hf[s] - Hb * z2[s] <= 0 for s in rg_all(S)), name="BndFill")
        constrs["BndCut"] = mo.addConstrs((hc[s] - Ht * z2[s] <= 0 for s in rg_all(S)), name="BndCut")
        constrs["BndPileLB"] = mo.addConstrs((Hb * z3[s] - hb[s] <= 0 for s in rg_all(S)), name="BndPileLB")
        constrs["BndPileUB"] = mo.addConstrs((hb[s] - (ema-emi) * z3[s] <= 0 for s in rg_all(S)), name="BndPileUB")
    
    if is_soc_included:
        constrs["SOCFill"] = gp.tupledict()
        constrs["SOCCut"] = gp.tupledict()
        constrs["SOCPile"] = gp.tupledict()
        for s in rg_all(S):
            constrs["SOCFill"][s] = mo.addQConstr(hf[s] ** 2 - hf2[s] <= 0, name=f"SOCFill[{s}]")
            constrs["SOCCut"][s] = mo.addQConstr(hc[s] ** 2 - hc2[s] <= 0, name=f"SOCCut[{s}]")
            constrs["SOCPile"][s] = mo.addQConstr(hb[s] ** 2 - hb2[s] <= 0, name=f"SOCPile[{s}]")
        
    if is_q_cost_included:
        constrs["CostEarth"] = mo.addConstrs(
            ((U_surface * z2[s] + U_cut * (We * hc[s] + Me * hc2[s]) + U_fill * (We * hf[s] + Me * hf2[s])) * ds / COST_SCALE - Ce[s] <= 0 
             for s in rg_all(S)), name="CostEarth")
        constrs["CostPile"] = mo.addConstrs(
            (-Cp[s] + c_pl * (hb[s] + r_b * hb2[s]) * ds / L_span / COST_SCALE <= 0 for s in rg_all(S)), name="CostPile")
    
    if is_l_cost_included:
        constrs["CostEarth"] = mo.addConstrs(
            ((U_surface * z2[s] + U_cut * (We + Me) * hc[s] + U_fill * (We + Me) * hf[s]) * ds / COST_SCALE - Ce[s] <= 0 
             for s in rg_all(S)), name="CostEarth")
        constrs["CostPile"] = mo.addConstrs(
            (-Cp[s] + c_pl * 2 * hb[s] * ds / L_span / COST_SCALE <= 0 for s in rg_all(S)), name="CostPile")
    
    mo.update()
    return constrs


def get_vao_objective(variables: dict[str, gp.tupledict] | dict[str, dict], S: int, scale: float) -> gp.LinExpr | float:
    """Get the objective expression for the VAO model."""
    Ce, Ct, Cb, Cp, Ctne = variables["Ce"], variables["Ct"], variables["Cb"], variables["Cp"], variables["Ctne"]
    cost = 0
    for s in rg_all(S):
        cost += Ce[s] + Ct[s] + Cb[s] + Cp[s] + Ctne[s]
    cost *= COST_SCALE * scale
    return cost


def build_vao_model(cfg: GroundConfigM2, scale: float | None = None, is_soc_included: bool = True) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build the full VAO M2 model."""
    model = gp.Model("vao")
    
    mp_vars = add_mp_variables(model, cfg)
    sp_vars = add_sp_variables(model, cfg)
    variables = {**mp_vars, **sp_vars}
    
    mp_cs = add_mp_constraints(model, cfg, mp_vars)
    sp_cs = add_sp_constraints(model, cfg, sp_vars, mp_vars, is_soc_included=is_soc_included)
    constrs = {**mp_cs, **sp_cs}
    
    scale = scale or cfg.obj_scale
    construction_cost = get_vao_objective(variables, cfg.S, scale)
    model.setObjective(construction_cost, gp.GRB.MINIMIZE)
    model.update()
    return model, variables, constrs


def generate_flat_track(cfg: GroundConfigM2, is_mid_vpi: bool = False) -> dict[str, dict[int, float]]:
    """Generate a flat track connecting directly between two platforms."""
    sigma = cfg.sigma
    hbar = dict(enumerate(cfg.e6g, start=1))
    S, ds = cfg.S, cfg.ds
    emi, ema = cfg.y_range
    S_plat = cfg.platform_intervals
    
    m = gp.Model("flat_track")
    
    e = m.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=emi, ub=ema, name="e")
    e0 = m.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=ema, name="e0")
    et = m.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=ema-cfg.ht_min, name="et")
    eb = m.addVars(rg_all(S), vtype=GRB.CONTINUOUS, lb=0, ub=ema, name="eb")
    pi = m.addVars(rg_all(S), vtype=GRB.BINARY, name="pi")
    z1 = m.addVars(rg_all(S), vtype=GRB.BINARY, name="z1")
    z2 = m.addVars(rg_all(S), vtype=GRB.BINARY, name="z2")
    z3 = m.addVars(rg_all(S), vtype=GRB.BINARY, name="z3")
    variables = {"e": e, "e0": e0, "et": et, "eb": eb, "pi": pi, "z1": z1, "z2": z2, "z3": z3}
    
    m.addConstrs((z1[s] + z2[s] + z3[s] == 1 for s in rg_all(S)), name="ModeSum")
    m.addConstrs((e[s] - et[s] - e0[s] - eb[s] == 0 for s in rg_all(S)), name="ElevDecomp")
    m.addConstrs((emi * z1[s] - et[s] <= 0 for s in rg_all(S)), name="ElevTunnelLB")
    m.addConstrs((et[s] - (hbar[s] - cfg.ht_min) * z1[s] <= 0 for s in rg_all(S)), name="ElevTunnelUB")
    m.addConstrs(((hbar[s] - cfg.ht_min) * z2[s] - e0[s] <= 0 for s in rg_all(S)), name="ElevEarthLB")
    m.addConstrs((e0[s] - (hbar[s] + cfg.hb_min) * z2[s] <= 0 for s in rg_all(S)), name="ElevEarthUB")
    m.addConstrs(((hbar[s] + cfg.hb_min) * z3[s] - eb[s] <= 0 for s in rg_all(S)), name="ElevBridgeLB")
    m.addConstrs((eb[s] - ema * z3[s] <= 0 for s in rg_all(S)), name="ElevBridgeUB")
    m.addConstrs((gp.quicksum(pi[s] for s in range(i, i+sigma)) <= 1 for i in range(1, S - sigma + 2)), name="VPISpacing")
    m.addConstrs((-cfg.di_max * ds * pi[s] - e[s+1] - e[s-1] + 2*e[s] <= 0 for s in rg_curv(S)), name="CurvatureLB")
    m.addConstrs((e[s + 1] + e[s - 1] - 2 * e[s] - cfg.di_max * ds * pi[s] <= 0 for s in rg_curv(S)), name="CurvatureUB")
    m.addConstrs((e[s] - e[s+1] - cfg.i_max * ds <= 0 for s in rg_grad(S)), name="GradientLB")
    m.addConstrs((e[s+1] - e[s] - cfg.i_max * ds <= 0 for s in rg_grad(S)), name="GradientUB")
    for i in rg_plat(S, S_plat):
        m.addConstr(e[i] == hbar[i], name=f"Platform[{i}]")
    
    if is_mid_vpi:
        mid_vpi_cs = m.addConstr(gp.quicksum(pi[s] for s in range(S_plat+2, S+2-S_plat-1)) == 1, name="flat_track_mid_VPI")
    else:
        for i in range(1 + S_plat + 1, S + 2 - S_plat - 1):
            m.addConstr(pi[i] == 0, name=f"flat_track_no_VPI_at[{i}]")
    
    m.setObjective(gp.quicksum((e[s] - hbar[s])**2 for s in rg_all(S)), gp.GRB.MINIMIZE)
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = 30.0
    m.Params.MIPGap = 0.05
    m.optimize()
    
    if m.status == gp.GRB.OPTIMAL or m.SolCount >= 1:
        return {varb: {k: v.X for k, v in variables[varb].items()} for varb in variables}
    
    if m.status == gp.GRB.TIME_LIMIT and m.SolCount == 0:
        print("[Warning] Optimization timed out, falling back to feasibility search...")
        if is_mid_vpi:
            m.remove(mid_vpi_cs)  # type: ignore
            for i in range(1 + S_plat + 1, S + 2 - S_plat - 1):
                m.addConstr(pi[i] == 0, name=f"fallback_no_VPI_at[{i}]")
        m.setObjective(0, gp.GRB.MINIMIZE)
        m.Params.TimeLimit = 30.0
        m.Params.SolutionLimit = 1
        m.optimize()
        if m.status == gp.GRB.OPTIMAL or m.SolCount >= 1:
            print("Fallback feasibility search succeeded.")
            return {varb: {k: v.X for k, v in variables[varb].items()} for varb in variables}
    
    raise RuntimeError(f"Flat track optimization failed with status: {STATUS_NAMES[m.status]}")


def build_vao_mp_model(cfg: GroundConfigM2, theta1_lb: float = -1e12) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build the VAO M2 Master-Problem."""
    model = gp.Model("vao_mp")
    
    mp_vars = add_mp_variables(model, cfg)
    theta1 = model.addVar(lb=theta1_lb, name="theta1")
    mp_vars['theta1'] = theta1  # type: ignore
    
    mp_cs = add_mp_constraints(model, cfg, mp_vars)
    
    Ct, Cb, Ctne = mp_vars["Ct"], mp_vars["Cb"], mp_vars["Ctne"]
    S = cfg.S
    fixed_cost = gp.quicksum(Ct[s] + Cb[s] + Ctne[s] for s in rg_all(S)) * COST_SCALE
    model.setObjective(fixed_cost + theta1, GRB.MINIMIZE)
    model.update()
    
    return model, mp_vars, mp_cs


def build_vao_sp_model(cfg: GroundConfigM2, mp_vars: dict[str, dict[int, float]]) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build the VAO M2 Sub-Problem."""
    model = gp.Model("vao_sp")
    
    sp_vars = add_sp_variables(model, cfg)
    sp_cs = add_sp_constraints(model, cfg, sp_vars, mp_vars)
    
    Cp, Ce = sp_vars["Cp"], sp_vars["Ce"]
    S = cfg.S
    variable_costs = gp.quicksum(Cp[s] + Ce[s] for s in rg_all(S)) * COST_SCALE
    model.setObjective(variable_costs, GRB.MINIMIZE)
    model.update()
    
    return model, sp_vars, sp_cs


def build_vao_sp_lp_model(cfg: GroundConfigM2, mp_vars: dict[str, dict[int, float]]) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build the VAO M2 Sub-Problem in LP relaxation form (no SOC constraints, linear cost)."""
    model = gp.Model("vao_sp_lp")
    
    sp_vars = add_sp_variables(model, cfg)
    sp_cs = add_sp_constraints(
        model, cfg, sp_vars, mp_vars, 
        is_height_included=True,
        is_soc_included=False, 
        is_q_cost_included=False, 
        is_l_cost_included=True, 
    )
    
    Cp, Ce = sp_vars["Cp"], sp_vars["Ce"]
    S = cfg.S
    variable_costs = gp.quicksum(Cp[s] + Ce[s] for s in rg_all(S)) * COST_SCALE
    model.setObjective(variable_costs, GRB.MINIMIZE)
    model.update()
    return model, sp_vars, sp_cs


def build_vao_sp_feasibility_model(cfg: GroundConfigM2, mp_vars: dict[str, dict[int, float]]) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build the VAO M2 Sub-Problem for feasibility check (used for generating FarkasDual)."""
    model = gp.Model("vao_sp_feas")
    
    sp_vars = add_sp_variables(model, cfg)
    sp_cs = add_sp_constraints(
        model, cfg, sp_vars, mp_vars, 
        is_soc_included=False, 
        is_q_cost_included=False, 
        is_l_cost_included=False, 
        is_height_included=False
    )
    
    model.setObjective(0, GRB.MINIMIZE)
    model.update()
    return model, sp_vars, sp_cs


def compute_feas_cut(
    sp_cs: dict[str, gp.tupledict],
    mp_vars: dict[str, gp.tupledict],
    gcfg: GroundConfigM2,
    is_zero_removed: bool = True,
    verbose: bool = False,
) -> gp.LinExpr | None:
    """Compute the feasibility cut constraint expression (LHS). Feas cut is: expr >= 0."""
    emi, ema = gcfg.y_range
    Ht, Hb = gcfg.ht_min, gcfg.hb_min
    S, ds = gcfg.S, gcfg.ds
    di, ima = gcfg.di_max, gcfg.i_max
    hbar = dict(enumerate(gcfg.e6g, start=1))
    S_plat = gcfg.platform_intervals

    pi, z1, z2, z3 = mp_vars["pi"], mp_vars["z1"], mp_vars["z2"], mp_vars["z3"]
    sp_farkas = extract_dual_info(sp_cs, GRB.INFEASIBLE, verbose=verbose)
    if sp_farkas is None:
        return None
    
    ETL, ETU = sp_farkas["ElevTunnelLB"], sp_farkas["ElevTunnelUB"]
    EEL, EEU = sp_farkas["ElevEarthLB"], sp_farkas["ElevEarthUB"]
    EBL, EBU = sp_farkas["ElevBridgeLB"], sp_farkas["ElevBridgeUB"]
    CL, CU = sp_farkas["CurvatureLB"], sp_farkas["CurvatureUB"]
    GL, GB = sp_farkas["GradientLB"], sp_farkas["GradientUB"]
    P = sp_farkas["Platform"]

    z1_coef = lambda s: -emi * ETL[s] + (hbar[s] - Ht) * ETU[s]
    z2_coef = lambda s: -(hbar[s] - Ht) * EEL[s] + (hbar[s] + Hb) * EEU[s]
    z3_coef = lambda s: -(hbar[s] + Hb) * EBL[s] + ema * EBU[s]
    pi_coef = lambda s: di * ds * (CL[s] + CU[s])
    
    cut_expr = gp.quicksum(ima * ds * (GL[s] + GB[s]) for s in rg_grad(S)) + gp.quicksum(P[s] * hbar[s] for s in rg_plat(S, S_plat))

    is_nonzero = lambda coeff: abs(coeff) > 1e-10
    
    if not is_zero_removed:
        cut_expr += gp.quicksum(
            z1_coef(s) * z1[s] + z2_coef(s) * z2[s] + z3_coef(s) * z3[s]
            for s in rg_all(S)
        ) + gp.quicksum(pi_coef(s) * pi[s] for s in rg_curv(S))
    else:
        cut_expr += gp.quicksum(
            (z1_coef(s) * z1[s] if is_nonzero(z1_coef(s)) else 0) +
            (z2_coef(s) * z2[s] if is_nonzero(z2_coef(s)) else 0) +
            (z3_coef(s) * z3[s] if is_nonzero(z3_coef(s)) else 0)
            for s in rg_all(S)
        ) + gp.quicksum(
            (pi_coef(s) * pi[s] if is_nonzero(pi_coef(s)) else 0)
            for s in rg_curv(S)
        )
    return cut_expr


def compute_optim_cut(
    sp_cs: dict[str, gp.tupledict],
    mp_vars: dict[str, gp.tupledict],
    gcfg: GroundConfigM2,
    sp_objval: float,
    mp_vars_vals: dict[str, dict[int, float]],
    is_zero_removed: bool = True,
    verbose: bool = False,
) -> gp.LinExpr | None:
    """Compute the optimality cut constraint expression. Optimality cut is: theta1 >= expr."""
    emi, ema = gcfg.y_range
    Ht, Hb = gcfg.ht_min, gcfg.hb_min
    S, ds = gcfg.S, gcfg.ds
    di = gcfg.di_max
    hbar = dict(enumerate(gcfg.e6g, start=1))
    S_plat = gcfg.platform_intervals
    U_surface = gcfg.U_surface
    Hmax = ema - emi

    pi, z1, z2, z3 = mp_vars["pi"], mp_vars["z1"], mp_vars["z2"], mp_vars["z3"]
    pi_k, z1_k, z2_k, z3_k = mp_vars_vals["pi"], mp_vars_vals["z1"], mp_vars_vals["z2"], mp_vars_vals["z3"]
    
    sp_dual = extract_dual_info(sp_cs, GRB.OPTIMAL, verbose=verbose)
    if sp_dual is None:
        return None
    
    def _get(name: str):
        return sp_dual.get(name, {})
    
    ETL, ETU = _get("ElevTunnelLB"), _get("ElevTunnelUB")
    EEL, EEU = _get("ElevEarthLB"), _get("ElevEarthUB")
    EBL, EBU = _get("ElevBridgeLB"), _get("ElevBridgeUB")
    CL, CU = _get("CurvatureLB"), _get("CurvatureUB")
    HE, HP = _get("HeightEarth"), _get("HeightBridge")
    BF, BC = _get("BndFill"), _get("BndCut")
    BPL, BPU = _get("BndPileLB"), _get("BndPileUB")
    CE, CP = _get("CostEarth"), _get("CostPile")
    
    z2_base_coef = lambda s: (
        (hbar[s] + Hb) * EEU[s] - (hbar[s] - Ht) * EEL[s] + hbar[s] * HE[s] +
        Hb * BF[s] + Ht * BC[s] - ds * U_surface / COST_SCALE * CE[s]
    )
    z3_base_coef = lambda s: (
        ema * EBU[s] - (hbar[s] + Hb) * EBL[s] + hbar[s] * HP[s] + Hmax * BPU[s] - Hb * BPL[s]
    )
    pi_coef = lambda s: di * ds * (CL[s] + CU[s])
    z1_coef = lambda s: (hbar[s] - Ht) * ETU[s] - emi * ETL[s]
    z2_coef = lambda s: z2_base_coef(s)
    z3_coef = lambda s: z3_base_coef(s)

    is_nonzero = lambda coeff: abs(coeff) > 1e-10
    
    cut_expr = gp.LinExpr(sp_objval)
    for s in rg_all(S):
        c1, c2, c3 = z1_coef(s), z2_coef(s), z3_coef(s)
        if not is_zero_removed or is_nonzero(c1):
            cut_expr += c1 * (z1[s] - z1_k[s])
        if not is_zero_removed or is_nonzero(c2):
            cut_expr += c2 * (z2[s] - z2_k[s])
        if not is_zero_removed or is_nonzero(c3):
            cut_expr += c3 * (z3[s] - z3_k[s])
    for s in rg_curv(S):
        cp = pi_coef(s)
        if not is_zero_removed or is_nonzero(cp):
            cut_expr += cp * (pi[s] - pi_k[s])
    return cut_expr


def compute_nogood_cut(
    mp_vars: dict[str, gp.tupledict],
    mp_vars_vals: dict[str, dict[int, float]],
    gcfg: GroundConfigM2,
    sp_objval: float,
) -> gp.LinExpr:
    """Compute a no-good cut when standard Benders cut fails."""
    S = gcfg.S
    
    z1, z2, z3, pi = mp_vars["z1"], mp_vars["z2"], mp_vars["z3"], mp_vars["pi"]
    z1_k, z2_k, z3_k, pi_k = mp_vars_vals["z1"], mp_vars_vals["z2"], mp_vars_vals["z3"], mp_vars_vals["pi"]
    
    disagreement_expr = gp.LinExpr()
    
    for s in rg_all(S):
        disagreement_expr += (1 - z1[s]) if z1_k[s] > 0.5 else z1[s]
        disagreement_expr += (1 - z2[s]) if z2_k[s] > 0.5 else z2[s]
        disagreement_expr += (1 - z3[s]) if z3_k[s] > 0.5 else z3[s]
    
    for s in rg_curv(S):
        disagreement_expr += (1 - pi[s]) if pi_k[s] > 0.5 else pi[s]
    
    M = max(sp_objval, 1.0)
    cut_rhs = sp_objval - M * disagreement_expr
    
    return cut_rhs


def compute_simple_nogood_cut(
    mp_vars: dict[str, gp.tupledict],
    mp_vars_vals: dict[str, dict[int, float]],
    gcfg: GroundConfigM2,
) -> gp.LinExpr:
    """Compute a simple no-good cut that excludes the current binary solution. Cut is: expr >= 1."""
    S = gcfg.S
    
    z1, z2, z3, pi = mp_vars["z1"], mp_vars["z2"], mp_vars["z3"], mp_vars["pi"]
    z1_k, z2_k, z3_k, pi_k = mp_vars_vals["z1"], mp_vars_vals["z2"], mp_vars_vals["z3"], mp_vars_vals["pi"]
    
    cut_lhs = gp.LinExpr()
    
    for s in rg_all(S):
        cut_lhs += (1 - z1[s]) if z1_k[s] > 0.5 else z1[s]
        cut_lhs += (1 - z2[s]) if z2_k[s] > 0.5 else z2[s]
        cut_lhs += (1 - z3[s]) if z3_k[s] > 0.5 else z3[s]
    
    for s in rg_curv(S):
        cut_lhs += (1 - pi[s]) if pi_k[s] > 0.5 else pi[s]
    
    return cut_lhs


@dataclass
class OACut:
    """An Outer Approximation Cut. For SOC constraint h^2 <= t, the cut at (h*, t*) is: 2*h* * h - t <= h*^2."""
    h_var: gp.Var
    t_var: gp.Var
    coeff_h: float
    coeff_t: float
    rhs: float
    name: str
    h_val: float
    violation: float
    
    def get_expr(self) -> gp.LinExpr:
        return self.coeff_h * self.h_var + self.coeff_t * self.t_var - self.rhs


def check_soc_violation(h_val: float, t_val: float) -> float:
    return h_val ** 2 - t_val


def generate_single_oa_cut(
    h_var: gp.Var,
    t_var: gp.Var, 
    h_val: float,
    t_val: float,
    name: str,
    min_h_for_cut: float = 1e-6
) -> OACut | None:
    """Generate OA cut for a single SOC constraint h^2 <= t at point (h*, t*)."""
    violation = check_soc_violation(h_val, t_val)
    
    if violation <= 1e-8 or abs(h_val) < min_h_for_cut:
        return None
    
    coeff_h = 2.0 * h_val
    coeff_t = -1.0
    rhs = h_val ** 2
    
    return OACut(
        h_var=h_var, t_var=t_var,
        coeff_h=coeff_h, coeff_t=coeff_t, rhs=rhs,
        name=name, h_val=h_val, violation=violation
    )


def compute_initial_oa_cuts(
    sp_vars: dict[str, gp.tupledict],
    gcfg: GroundConfigM2,
    min_h_for_cut: float = 1e-6,
) -> list[OACut]:
    """Compute initial OA cuts at reference points."""
    cuts = []
    S = gcfg.S
    Hb, Ht = gcfg.hb_min, gcfg.ht_min
    H_max = gcfg.y_range[1] - gcfg.y_range[0]
    
    hf, hc, hb = sp_vars["hf"], sp_vars["hc"], sp_vars["hb"]
    hf2, hc2, hb2 = sp_vars["hf2"], sp_vars["hc2"], sp_vars["hb2"]
    
    for s in rg_all(S):
        ref_hf = Hb / 2
        if ref_hf > min_h_for_cut:
            cuts.append(OACut(h_var=hf[s], t_var=hf2[s], coeff_h=2.0*ref_hf, coeff_t=-1.0, 
                              rhs=ref_hf**2, name=f"InitOAFill[{s}]", h_val=ref_hf, violation=0))
        
        ref_hc = Ht / 2
        if ref_hc > min_h_for_cut:
            cuts.append(OACut(h_var=hc[s], t_var=hc2[s], coeff_h=2.0*ref_hc, coeff_t=-1.0,
                              rhs=ref_hc**2, name=f"InitOACut[{s}]", h_val=ref_hc, violation=0))
        
        ref_hb = H_max / 2
        if ref_hb > min_h_for_cut:
            cuts.append(OACut(h_var=hb[s], t_var=hb2[s], coeff_h=2.0*ref_hb, coeff_t=-1.0,
                              rhs=ref_hb**2, name=f"InitOAPile[{s}]", h_val=ref_hb, violation=0))
    
    return cuts


def compute_oa_cuts_from_solution(
    sp_vars: dict[str, gp.tupledict],
    cfg,
    min_violation: float = 1e-6,
    min_h_for_cut: float = 1e-6,
) -> list[OACut]:
    """Compute OA cuts from subproblem solution."""
    cuts = []
    S = cfg.S
    
    hf, hc, hb = sp_vars["hf"], sp_vars["hc"], sp_vars["hb"]
    hf2, hc2, hb2 = sp_vars["hf2"], sp_vars["hc2"], sp_vars["hb2"]
    
    for s in rg_all(S):
        cut = generate_single_oa_cut(hf[s], hf2[s], hf[s].X, hf2[s].X, f"OAFill[{s}]", min_h_for_cut)
        if cut is not None and cut.violation > min_violation:
            cuts.append(cut)
        
        cut = generate_single_oa_cut(hc[s], hc2[s], hc[s].X, hc2[s].X, f"OACut[{s}]", min_h_for_cut)
        if cut is not None and cut.violation > min_violation:
            cuts.append(cut)
        
        cut = generate_single_oa_cut(hb[s], hb2[s], hb[s].X, hb2[s].X, f"OAPile[{s}]", min_h_for_cut)
        if cut is not None and cut.violation > min_violation:
            cuts.append(cut)
    
    return cuts


def compute_oa_cuts_from_values(
    h_vals: dict[str, dict[int, float]],
    t_vals: dict[str, dict[int, float]],
    h_vars: dict[str, gp.tupledict],
    t_vars: dict[str, gp.tupledict],
    cfg,
    min_violation: float = 1e-6,
    min_h_for_cut: float = 1e-6,
) -> list[OACut]:
    """Compute OA cuts from variable values (for use in callbacks)."""
    cuts = []
    S = cfg.S
    
    h_t_pairs = [("hf", "hf2", "OAFill"), ("hc", "hc2", "OACut"), ("hb", "hb2", "OAPile")]
    
    for h_name, t_name, cut_prefix in h_t_pairs:
        for s in rg_all(S):
            cut = generate_single_oa_cut(
                h_vars[h_name][s], t_vars[t_name][s],
                h_vals[h_name][s], t_vals[t_name][s],
                f"{cut_prefix}[{s}]", min_h_for_cut
            )
            if cut is not None and cut.violation > min_violation:
                cuts.append(cut)
    
    return cuts


def add_oa_cut_to_model(model: gp.Model, cut: OACut) -> gp.Constr:
    """Add OA cut to model."""
    cs = model.addConstr(cut.get_expr() <= 0, name=cut.name)
    model.update()
    return cs


def add_oa_cut_lazy(model: gp.Model, cut: OACut):
    """Add OA cut as lazy constraint in callback."""
    model.cbLazy(cut.get_expr() <= 0)
    model.update()


def build_vao_oa_master_model(cfg: GroundConfigM2, theta1_lb: float = 0) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build OA master problem model (includes SP variables but no SOC constraints)."""
    model = gp.Model("vao_oa_master")
    
    mp_vars = add_mp_variables(model, cfg)
    sp_vars = add_sp_variables(model, cfg)
    all_vars = {**mp_vars, **sp_vars}
    
    mp_cs = add_mp_constraints(model, cfg, mp_vars)
    sp_cs = add_sp_constraints(
        model, cfg, sp_vars, mp_vars,
        is_height_included=True,
        is_soc_included=False,
        is_q_cost_included=True,
        is_l_cost_included=False,
    )
    
    all_cs = {**mp_cs, **sp_cs}
    
    obj = get_vao_objective(all_vars, cfg.S, scale=cfg.obj_scale)
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    
    return model, all_vars, all_cs


def build_vao_oa_subproblem(cfg, mp_sol: dict[str, dict[int, float]]) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build OA subproblem (SOCP with fixed integer variables)."""
    return build_vao_sp_model(cfg, mp_sol)


if __name__ == "__main__":
    from configs import GroundConfigM2
    
    gcfg = GroundConfigM2("HP3")
    vao_flat_sol = generate_flat_track(gcfg, True)