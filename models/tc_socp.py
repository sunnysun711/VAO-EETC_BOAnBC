"""Train Control Second-Order Cone Programming (SOCP) Model.

This module implements the energy-efficient train control model with
convexified kinematics constraints.
"""

from pathlib import Path
from typing import Any, Callable, Optional
import gurobipy as gp

from configs import TrainConfig
from configs.ground import GroundConfigM2
from utils import STATUS_NAMES
from utils.gurobi import extract_dual_info, extract_gp_solution


FORCE_SCALE = 1000.0  # N to kN
MIN_V_EPS = 1e-4  # m/s
MIN_U_EPS = 1e-2  # kN
USE_ACTIVE_CONSTRAINTS_LP = False


def add_tc_variables(
    model: gp.Model,
    cfg: TrainConfig,
    S: int,
) -> dict[str, gp.tupledict]:
    """Add train control model variables."""
    mass = cfg.mass
    max_tra_f = cfg.max_tra_force
    max_bra_f = cfg.max_reg_bra_force
    max_v = cfg.max_v
    min_v = cfg.min_v
    r0, r1, r2 = cfg.r0, cfg.r1, cfg.r2
    r_tn = cfg.rtn

    a_max = cfg.max_acc
    w0_max = r0 + r1 * max_v + r2 * max_v**2
    wtn_max = r_tn * max_v**2
    wi_max = cfg.i_max * mass * cfg.g

    uPlus = model.addVars(range(1, S + 1), lb=0, ub=max_tra_f / FORCE_SCALE, name=f"uPlus")
    uMinus = model.addVars(range(1, S + 1), lb=0, ub=max_bra_f / FORCE_SCALE, name=f"uMinus")
    v = model.addVars(range(1, S + 2), lb=min_v, ub=max_v, name=f"v")
    a = model.addVars(range(1, S + 1), lb=-a_max, ub=a_max, name=f"a")
    alpha = model.addVars(range(1, S + 2), lb=1 / max_v, ub=1 / min_v, name=f"alpha")
    beta = model.addVars(range(1, S + 2), lb=min_v**2, ub=max_v**2, name=f"beta")
    w0 = model.addVars(range(1, S + 1), lb=0, ub=w0_max / FORCE_SCALE, name=f"w0")
    wtn = model.addVars(range(1, S + 1), lb=0, ub=wtn_max / FORCE_SCALE, name=f"wtn")
    wi = model.addVars(range(1, S + 1), lb=-wi_max / FORCE_SCALE, ub=wi_max / FORCE_SCALE, name=f"wi")

    model.update()

    variables = {
        "uPlus": uPlus,
        "uMinus": uMinus,
        "v": v,
        "a": a,
        "alpha": alpha,
        "beta": beta,
        "w0": w0,
        "wtn": wtn,
        "wi": wi,
    }
    return variables


def add_tc_constraints(
    model: gp.Model,
    cfg: TrainConfig,
    vao_vars: dict[str, gp.tupledict | Any] | None,
    variables: dict[str, gp.tupledict],
    S: int,
    ds: float,
    is_uphill_dir: bool = True,
    is_time_constrained: bool = True,
    is_coupling_included: bool = True,
    is_soc_included: bool = True,
    cs_name_prefix: str = "",
    tighten_time_limit: float = 0.0,
) -> dict[str, gp.tupledict]:
    """Add train control constraints to the model."""
    uPlus  = variables["uPlus"]
    uMinus = variables["uMinus"]
    v      = variables["v"]
    a      = variables["a"]
    alpha  = variables["alpha"]
    beta   = variables["beta"]
    w0     = variables["w0"]
    wtn    = variables["wtn"]
    wi     = variables["wi"]

    r0, r1, r2 = cfg.r0, cfg.r1, cfg.r2
    mass, g, rho = cfg.mass, cfg.g, cfg.rho

    direction_multiplier = 1 if is_uphill_dir else -1
    npfx = cs_name_prefix

    constrs: dict[str, gp.tupledict] = {}

    _c = cfg.max_tra_power / FORCE_SCALE
    constrs["u1_power"] = model.addConstrs(
        (uPlus[s] - _c * alpha[s] <= 0 for s in range(1, S+1)),
        name=f"{npfx}u1_power"
    )
    for constraint in cfg.max_tra_cs:
        params = constraint["params"]
        if constraint["type"] == "linear":
            for i in range(len(params)):
                constrs[f"u1_linear{i+1}"] = model.addConstrs(
                    (uPlus[s] - params[i]["k"] * v[s] <= params[i]["b"] for s in range(1, S+1)),
                    name=f"{npfx}u1_linear{i+1}"
                )
        elif constraint["type"] == "inverse":
            for i in range(len(params)):
                constrs[f"u1_inverse{i+1}"] = model.addConstrs(
                    (uPlus[s] - params[i]["a"] * alpha[s] <= params[i]["c"] for s in range(1, S+1)),
                    name=f"{npfx}u1_inverse{i+1}"
                )

    _c = cfg.max_reg_bra_power / FORCE_SCALE
    constrs["u2_power"] = model.addConstrs(
        (uMinus[s] - _c * alpha[s] <= 0 for s in range(1, S+1)),
        name=f"{npfx}u2_power"
    )
    for constraint in cfg.max_bra_cs:
        params = constraint["params"]
        if constraint["type"] == "linear":
            for i in range(len(params)):
                constrs[f"u2_linear{i+1}"] = model.addConstrs(
                    (uMinus[s] - params[i]["k"] * v[s] <= params[i]["b"] for s in range(1, S+1)),
                    name=f"{npfx}u2_linear{i+1}"
                )
        elif constraint["type"] == "inverse":
            for i in range(len(params)):
                constrs[f"u2_inverse{i+1}"] = model.addConstrs(
                    (uMinus[s] - params[i]["a"] * alpha[s] <= params[i]["c"] for s in range(1, S+1)),
                    name=f"{npfx}u2_inverse{i+1}"
                )

    if is_soc_included:
        constrs.update(add_tc_soc_constraints(model, S, variables, npfx))

    constrs["w0"] = model.addConstrs(
        (FORCE_SCALE * w0[s] - r1 * v[s] - r2 * beta[s] == r0 for s in range(1, S+1)),
        name=f"{npfx}w0"
    )

    if is_coupling_included and vao_vars is not None:
        coupling_constrs = add_tc_coupling_constraints(model, cfg, vao_vars, variables, S, ds, is_uphill_dir, npfx)
        if coupling_constrs is not None:
            constrs.update(coupling_constrs)

    constrs["dvdt"] = model.addConstrs(
        (a[s] - direction_multiplier / 2 / ds * (beta[s+1] - beta[s]) == 0 for s in range(1, S+1)),
        name=f"{npfx}dvdt"
    )

    _c = mass * (1+rho) / FORCE_SCALE
    constrs["newton"] = model.addConstrs(
        (a[s] * _c - (uPlus[s] - uMinus[s] - w0[s] - wtn[s] - wi[s]) == 0 for s in range(1, S+1)),
        name=f"{npfx}newton"
    )

    if is_time_constrained:
        constrs["time"] = gp.tupledict()
        constrs["time"][None] = add_tc_time_constraint(
            model, S, ds, alpha,
            running_time_max= cfg.T - tighten_time_limit,
            cs_name_prefix=npfx,
        )

    constrs["v_bound"] = gp.tupledict()
    constrs["beta_bound"] = gp.tupledict()
    constrs["v_bound"][1] = model.addConstr((v[1] == cfg.min_v), name=f"{npfx}v1")
    constrs["v_bound"][S+1] = model.addConstr((v[S+1] == cfg.min_v), name=f"{npfx}vS1")
    constrs["beta_bound"][1] = model.addConstr((beta[1] == cfg.min_v**2), name=f"{npfx}beta1")
    constrs["beta_bound"][S+1] = model.addConstr((beta[S+1] == cfg.min_v**2), name=f"{npfx}betaS1")

    model.update()
    return constrs


def add_tc_time_constraint(
    md: gp.Model,
    S: int,
    ds: float,
    alpha: gp.tupledict,
    running_time_max: float,
    cs_name_prefix: str = ""
) -> gp.Constr:
    """Add time constraint to the model."""
    return md.addConstr(
        gp.quicksum(alpha[s] for s in range(1, S+1)) * ds <= running_time_max,
        name=f"{cs_name_prefix}time"
    )


def add_tc_soc_constraints(
    md: gp.Model,
    S: int,
    variables: dict[str, gp.tupledict],
    cs_name_prefix: str = "",
) -> dict[str, gp.tupledict]:
    """Add SOC relaxation constraints: beta >= v^2, alpha * v >= 1."""
    beta = variables["beta"]
    alpha = variables["alpha"]
    v = variables["v"]

    constrs = {}
    constrs["beta_relax"] = gp.tupledict()
    constrs["alpha_relax"] = gp.tupledict()
    for i in range(1, S+2):
        constrs["beta_relax"][i] = md.addQConstr(
            (beta[i] >= v[i]**2),
            name=f"{cs_name_prefix}beta_relax[{i}]"
        )
        constrs["alpha_relax"][i] = md.addQConstr(
            (alpha[i] * v[i] >= 1),
            name=f"{cs_name_prefix}alpha_relax[{i}]"
        )
    return constrs


def add_tc_coupling_constraints(
    md: gp.Model,
    cfg: TrainConfig,
    vao_vars: dict[str, gp.tupledict] | dict[str, dict[Any, float]],
    variables: dict[str, gp.tupledict],
    S: int,
    ds: float,
    is_uphill_dir: bool = True,
    cs_name_prefix: str = "",
) -> dict[str, gp.tupledict]:
    """Add coupling constraints of wi and wtn."""
    mass, g = cfg.mass, cfg.g
    rtn, max_v = cfg.rtn, cfg.max_v
    dir_mul = 1 if is_uphill_dir else -1

    e, z1 = vao_vars["e"], vao_vars["z1"]

    wi = variables["wi"]
    wtn = variables["wtn"]
    beta = variables["beta"]

    diff_e_coef = dir_mul * mass * g / ds / FORCE_SCALE
    c_wi = md.addConstrs((wi[s] == diff_e_coef * (e[s + 1] - e[s]) for s in range(1, S + 1)), name=f"{cs_name_prefix}wi",)
    c_wtn_l1 = md.addConstrs(( wtn[s] <= rtn * max_v**2 / FORCE_SCALE * z1[s] for s in range(1, S + 1)), name=f"{cs_name_prefix}wtn_linear1")
    c_wtn_l2 = md.addConstrs(( wtn[s] - rtn / FORCE_SCALE * beta[s] <= 0 for s in range(1, S + 1)), name=f"{cs_name_prefix}wtn_linear2")
    c_wtn_l3 = md.addConstrs((-wtn[s] + rtn / FORCE_SCALE * beta[s] <= rtn * max_v**2 / FORCE_SCALE * (1 - z1[s]) for s in range(1, S + 1)), name=f"{cs_name_prefix}wtn_linear3",)

    md.update()
    return {
        "wi": c_wi,
        "wtn_linear1": c_wtn_l1,
        "wtn_linear2": c_wtn_l2,
        "wtn_linear3": c_wtn_l3,
    }


def update_tc_coupling_constraints(
    model: gp.Model,
    variables: dict[str, gp.tupledict],
    constraints: dict[str, gp.tupledict | gp.Constr],
    vao_sol: dict[str, dict[Any, float]],
    gcfg: GroundConfigM2,
    tcfg: TrainConfig,
    is_uphill_dir: bool = True,
    cs_name_prefix: str = "",
) -> dict[str, gp.tupledict | gp.Constr]:
    """Update coupling constraints with given vao solution values."""
    for cs_name in ["wi", "wtn_linear1", "wtn_linear2", "wtn_linear3"]:
        if cs_name in constraints:
            model.remove(constraints[cs_name])
    new_cs = add_tc_coupling_constraints(
            model, tcfg, vao_sol, variables, S=gcfg.S, ds=gcfg.ds, is_uphill_dir=is_uphill_dir, cs_name_prefix=cs_name_prefix,
        )
    constraints.update(new_cs)
    return constraints


def add_eetc_objective(
    model: gp.Model,
    cfg: TrainConfig,
    variables: dict[str, gp.tupledict],
    S: int,
    ds: float,
    scale: float = 1.0,
    is_reg_included: bool = True,
) -> gp.LinExpr:
    """Add energy-efficient train control objective."""
    uPlus, uMinus = variables["uPlus"], variables["uMinus"]
    eta_tra, eta_bra = cfg.eff_tra, cfg.eff_reg_bra
    reg_weight = 1 if is_reg_included else 0
    new_obj = FORCE_SCALE * ds * gp.quicksum(
        1 / eta_tra * uPlus[s] - reg_weight * eta_bra * uMinus[s]
        for s in range(1, S + 1)
    )
    new_obj *= scale
    try:
        existing_obj = model.getObjective()
        model.setObjective(existing_obj + new_obj, gp.GRB.MINIMIZE)
    except gp.GurobiError:
        print("[WARN] No existing objective found. Setting new objective.")
        model.setObjective(new_obj, gp.GRB.MINIMIZE)
    model.update()
    return new_obj


def add_sotc_objective(
    model: gp.Model,
    cfg: TrainConfig,
    variables: dict[str, gp.tupledict],
    S: int,
    ds: float,
    scale: float = 1.0,
) -> gp.LinExpr:
    """Add speed-optimal train control objective (minimize travel time)."""
    new_obj = scale * ds * gp.quicksum(variables["alpha"][i] for i in range(1, S + 1))
    try:
        existing_obj = model.getObjective()
        model.setObjective(existing_obj + new_obj, gp.GRB.MINIMIZE)
    except gp.GurobiError:
        print("[WARN] No existing objective found. Setting new objective.")
        model.setObjective(new_obj, gp.GRB.MINIMIZE)
    model.update()
    return new_obj


def build_tc_model(
    cfg: TrainConfig,
    S: int,
    ds: float,
    is_uphill_dir: bool = True,
    vao_vars: dict[str, gp.tupledict | Any] = {},
    is_reg_included: bool = True,
    obj_scale: float = 1.,
    cs_name_prefix: str = "",
    tighten_time_limit: float = 0.0,
) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build the TC SOCP model."""
    model = gp.Model("tc_socp")
    variables = add_tc_variables(model, cfg, S=S)
    constraints = add_tc_constraints(
        model, cfg, vao_vars, variables, S=S, ds=ds, is_uphill_dir=is_uphill_dir,
        cs_name_prefix=cs_name_prefix, tighten_time_limit=tighten_time_limit,
    )
    add_eetc_objective(
        model, cfg, variables,
        S=S, ds=ds,
        is_reg_included=is_reg_included,
        scale=obj_scale,
    )
    return model, variables, constraints


def check_soc_active_status(
    socp_solution: dict[str, dict[int, float]],
    S: int,
    active_tol: float = 1e-5,
) -> tuple[dict[str, dict[int, bool]], list[str]]:
    """Check active status of SOC constraints.

    :return: (active_flags, log_messages)
    """
    v, beta, alpha = socp_solution["v"], socp_solution["beta"], socp_solution["alpha"]

    active = {"beta": {}, "alpha": {}}
    beta_inactive, alpha_inactive = [], []

    for i in range(1, S + 2):
        beta_slack = beta[i] - v[i] ** 2
        alpha_slack = alpha[i] * v[i] - 1

        active["beta"][i] = beta_slack < active_tol
        active["alpha"][i] = alpha_slack < active_tol

        if not active["beta"][i]:
            beta_inactive.append((i, beta_slack))
        if not active["alpha"][i]:
            alpha_inactive.append((i, alpha_slack))

    logs = []
    if beta_inactive:
        logs.append(f"beta_relax: {len(beta_inactive)}/{S+1} NOT active, max_slack={max(s for _,s in beta_inactive):.2e}")
    if alpha_inactive:
        logs.append(f"alpha_relax: {len(alpha_inactive)}/{S+1} NOT active, max_slack={max(s for _,s in alpha_inactive):.2e}")
    if not logs:
        logs.append("All SOC constraints are active")

    return active, logs


def build_active_constraints_lp(
    cfg: TrainConfig,
    socp_solution: dict[str, dict[int, float]],
    S: int,
    ds: float,
    is_uphill_dir: bool = True,
    vao_vars: dict[str, Any] = {},
    is_reg_included: bool = True,
    active_tol: float = 1e-5,
    cs_name_prefix: str = "",
    obj_scale: float = 1.0,
) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict], list[str]]:
    """Build LP model based on active SOC constraints."""
    active, logs = check_soc_active_status(socp_solution, S, active_tol)

    model = gp.Model("tc_active_lp")
    variables = add_tc_variables(model, cfg, S=S)

    model.remove(variables["v"])
    variables["v"] = gp.tupledict(socp_solution["v"])
    v, beta, alpha = variables["v"], variables["beta"], variables["alpha"]

    constraints = add_tc_constraints(
        model, cfg, vao_vars, variables, S=S, ds=ds,
        is_uphill_dir=is_uphill_dir,
        is_time_constrained=False,
        is_soc_included=False,
        cs_name_prefix=cs_name_prefix,
    )

    constraints["beta_active"] = gp.tupledict()
    constraints["alpha_active"] = gp.tupledict()

    for i in range(1, S + 2):
        if active["beta"][i]:
            constraints["beta_active"][i] = model.addConstr(
                beta[i] == v[i] ** 2, name=f"{cs_name_prefix}beta_eq[{i}]"
            )
        if active["alpha"][i]:
            constraints["alpha_active"][i] = model.addConstr(
                alpha[i] == 1.0 / v[i], name=f"{cs_name_prefix}alpha_eq[{i}]"
            )

    add_eetc_objective(model, cfg, variables, S=S, ds=ds, is_reg_included=is_reg_included, scale=obj_scale)
    model.update()

    return model, variables, constraints, logs


def build_tc_fixed_model(
    cfg: TrainConfig,
    socp_sol: dict[str, dict[int, float]],
    S: int,
    ds: float,
    is_uphill_dir: bool = True,
    vao_vars: dict[str, gp.tupledict | Any] = {},
    is_reg_included: bool = True,
    obj_scale: float = 1.,
    cs_name_prefix: str = "",
) -> tuple[gp.Model, dict[str, gp.tupledict], dict[str, gp.tupledict]]:
    """Build EETC-Fixed model (LP) with v, alpha, beta fixed from SOCP solution."""
    model = gp.Model("tc_fixed_lp")
    variables = add_tc_variables(model, cfg, S=S)

    for _k in ['v', 'alpha', 'beta']:
        model.remove(variables[_k])
        variables[_k] = gp.tupledict(socp_sol[_k])

    constraints = add_tc_constraints(
        model, cfg, vao_vars, variables, S=S, ds=ds, is_uphill_dir=is_uphill_dir,
        is_time_constrained=False,
        is_soc_included=False,
        cs_name_prefix=cs_name_prefix,
    )

    add_eetc_objective(model, cfg, variables, S=S, ds=ds, is_reg_included=is_reg_included, scale=obj_scale)
    return model, variables, constraints


def simulate_sotc(
    cfg: TrainConfig,
    vao_vars: dict[str, dict[int, float]],
    S: int,
    ds: float,
    is_uphill_dir: bool = True,
) -> float:
    """Simulate speed-optimal train control and return minimum travel time."""
    model = gp.Model("temp_tc_simulate")
    variables = add_tc_variables(model, cfg, S=S)
    add_tc_constraints(model, cfg, vao_vars, variables, S=S, ds=ds, is_uphill_dir=is_uphill_dir, is_time_constrained=False)
    add_sotc_objective(model, cfg, variables, S=S, ds=ds)
    model.setParam(gp.GRB.Param.OutputFlag, 0)
    model.optimize()
    if model.status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        raise ValueError(f"Model status is {model.status} {STATUS_NAMES[model.status]}, not optimal.")
    return model.ObjVal


def compute_optim_cut(
    sp_cs: dict[str, gp.tupledict],
    sub_obj_val: float,
    mp_vars: dict[str, gp.tupledict],
    mp_vars_bar: dict[str, dict[Any, float]],
    tcfg: TrainConfig,
    gcfg: GroundConfigM2,
    is_wtn_included: bool,
    is_uphill_dir: bool = True,
    is_zero_removed: bool = True,
) -> gp.LinExpr | None:
    """Compute the Benders optimality cut. Returns `theta >= LinExpr` or None if dual extraction fails."""
    S, ds = gcfg.S, gcfg.ds
    dir_mul = 1 if is_uphill_dir else -1
    M, g, max_v = tcfg.mass, tcfg.g, tcfg.max_v
    rtn = tcfg.rtn
    is_nonzero = lambda x: abs(x) > 1e-13

    sp_dual = extract_dual_info(sp_cs, gp.GRB.OPTIMAL)  # type: ignore
    if sp_dual is None:
        return None

    cut_expr = gp.LinExpr(sub_obj_val)

    assert "wi" in sp_cs, "wi must be included in sp_cs."
    assert "e" in mp_vars, "e must be included in mp_vars."
    assert "e" in mp_vars_bar, "e must be included in mp_vars_bar."
    e = mp_vars["e"]
    e_bar = mp_vars_bar["e"]
    WI = sp_dual["wi"]

    diff_e_coef = dir_mul * M * g / ds / FORCE_SCALE
    for s in range(1, S+1):
        c = WI[s] * diff_e_coef
        if not is_zero_removed or is_nonzero(c):
            cut_expr += c * ((e[s + 1] - e[s]) - (e_bar[s + 1] - e_bar[s]))

    if not is_wtn_included:
        return cut_expr

    assert "wtn_linear1" in sp_cs, "wtn_linear1 must be included in sp_cs."
    assert "wtn_linear3" in sp_cs, "wtn_linear3 must be included in sp_cs."
    assert "z1" in mp_vars, "z1 must be included in mp_vars."
    assert "z1" in mp_vars_bar, "z1 must be included in mp_vars_bar."

    z1 = mp_vars["z1"]
    z1_bar = mp_vars_bar["z1"]
    WTN1, WTN3 = sp_dual["wtn_linear1"], sp_dual["wtn_linear3"]

    for s in range(1, S+1):
        if not is_zero_removed or is_nonzero(WTN1[s]):
            cut_expr += WTN1[s] * (rtn * max_v**2 / FORCE_SCALE) * (z1[s] - z1_bar[s])
        if not is_zero_removed or is_nonzero(WTN3[s]):
            cut_expr += WTN3[s] * (rtn * max_v**2 / FORCE_SCALE) * ((-z1[s]) - (-z1_bar[s]))

    return cut_expr


def solve_tc_subproblem(
    tc_model: gp.Model,
    tcfg: TrainConfig,
    is_uphill_dir: bool,
    gcfg: GroundConfigM2,
    vao_sol: dict[str, dict[int, float]],
) -> tuple[gp.Model, Optional[float]]:
    """Solve TC subproblem. The tc_model must have _variables and _constraints attributes.

    :return: (tc_model, tc_obj_val)
    """
    update_tc_coupling_constraints(
        model=tc_model,
        variables=tc_model._variables,
        constraints=tc_model._constraints,
        vao_sol=vao_sol,
        gcfg=gcfg,
        tcfg=tcfg,
        is_uphill_dir=is_uphill_dir,
    )
    tc_model.optimize()

    if tc_model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        return tc_model, None

    return tc_model, tc_model.ObjVal


def _try_recover_infeasible_tc(
    tcfg: TrainConfig,
    gcfg: GroundConfigM2,
    vao_sol: dict[str, dict[Any, float]],
    mp_vars: dict[str, gp.tupledict],
    is_uphill_dir: bool,
    log_method: Callable,
    log_prefix: str,
    ti_scale: float,
    n_callback: int = 0,
    ti_idx: int = 0,
    run_dir: Path = Path("runs"),
) -> tuple[Optional[gp.LinExpr], str]:
    """Try to recover from infeasible TC subproblem by adding slack variables.

    :return: (cut_expr, model_str)
    """
    S, ds = gcfg.S, gcfg.ds
    LARGE_PENALTY = 1e12

    test_model = gp.Model("tc_test_time")
    test_vars = add_tc_variables(test_model, tcfg, S=S)
    add_tc_constraints(
        test_model, tcfg, vao_sol, test_vars, S=S, ds=ds,
        is_uphill_dir=is_uphill_dir,
        is_time_constrained=False,
    )
    add_sotc_objective(test_model, tcfg, test_vars, S=S, ds=ds, scale=1.0)
    test_model.Params.OutputFlag = 0
    test_model.optimize()

    time_is_issue = test_model.Status in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]

    slack_model = gp.Model("tc_slack")
    variables = add_tc_variables(slack_model, tcfg, S=S)

    if time_is_issue:
        log_method(f"{log_prefix}  Time constraint too tight ({test_model.ObjVal} > {tcfg.T}), adding T_slack.")

        constraints = add_tc_constraints(
            slack_model, tcfg, vao_sol, variables, S=S, ds=ds,
            is_uphill_dir=is_uphill_dir,
            is_time_constrained=False,
        )

        T_slack = slack_model.addVar(lb=0, name="T_slack")
        alpha = variables["alpha"]
        constraints["time"] = gp.tupledict()
        constraints["time"][None] = slack_model.addConstr(
            gp.quicksum(alpha[s] for s in range(1, S+1)) * ds - T_slack <= tcfg.T,
            name="time_with_slack"
        )

        slack_obj_no_penal = add_eetc_objective(
            slack_model, tcfg, variables, S=S, ds=ds, is_reg_included=False, scale=ti_scale)
        slack_model.setObjective(
            slack_model.getObjective() + LARGE_PENALTY * T_slack,
            gp.GRB.MINIMIZE
        )
        slack_type = "T"
    else:
        log_method(f"{log_prefix}  Traction insufficient, adding U_slack.")

        constraints = add_tc_constraints(
            slack_model, tcfg, vao_sol, variables, S=S, ds=ds,
            is_uphill_dir=is_uphill_dir,
            is_time_constrained=True,
        )

        U_slack_plus = slack_model.addVars(range(1, S+1), lb=0, name="U_slack_plus")
        U_slack_minus = slack_model.addVars(range(1, S+1), lb=0, name="U_slack_minus")

        slack_model.remove(constraints["newton"])
        _c = tcfg.mass * (1 + tcfg.rho) / FORCE_SCALE
        constraints["newton"] = slack_model.addConstrs(
            (variables["a"][s] * _c - (
                variables["uPlus"][s] + U_slack_plus[s] - U_slack_minus[s]
                - variables["uMinus"][s]
                - variables["w0"][s] - variables["wtn"][s] - variables["wi"][s]
            ) == 0 for s in range(1, S+1)),
            name="newton_with_slack"
        )

        slack_obj_no_penal = add_eetc_objective(
            slack_model, tcfg, variables, S=S, ds=ds, is_reg_included=False, scale=ti_scale)
        slack_model.setObjective(
            slack_model.getObjective()
            + LARGE_PENALTY
            * gp.quicksum(U_slack_plus[s] + U_slack_minus[s] for s in range(1, S + 1)),
            gp.GRB.MINIMIZE,
        )
        slack_type = "U"

    slack_model.update()
    slack_model.Params.OutputFlag = 0
    slack_model.optimize()

    if slack_model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        log_method(f"{log_prefix}  Slack model still {STATUS_NAMES.get(slack_model.Status, slack_model.Status)}!\t\t\t<== WARNING!")
        if slack_model.Status == gp.GRB.INFEASIBLE:
            slack_model.computeIIS()
            pat = run_dir / f"tc_slack_CB{n_callback}_TC{ti_idx}.ilp"
            pat.parent.mkdir(parents=True, exist_ok=True)
            slack_model.write(str(pat))
        return None, "slack_failed"

    slack_obj = slack_model.ObjVal
    slack_obj_no_penal = slack_obj_no_penal.getValue()
    log_method(f"{log_prefix}  Slack model solved, obj_no_penal={slack_obj_no_penal:.6e}, obj_penal={slack_obj:.6e}")

    cut_expr = compute_optim_cut(
        sp_cs=constraints,
        sub_obj_val=slack_obj_no_penal,  # no penalty
        mp_vars=mp_vars,
        mp_vars_bar=vao_sol,
        tcfg=tcfg,
        gcfg=gcfg,
        is_wtn_included=True,
        is_uphill_dir=is_uphill_dir,
    )

    if cut_expr is not None:
        log_method(f"{log_prefix}  Slack SOCP optim_cut extracted with obj_no_penal={slack_obj_no_penal:.6e}.")
        return cut_expr, f"slack_{slack_type}"

    log_method(f"{log_prefix}  Slack SOCP dual failed, try fixed-LP.")
    socp_sol = extract_gp_solution(variables)

    lp_model = gp.Model("tc_slack_fixed_lp")
    lp_vars = add_tc_variables(lp_model, tcfg, S=S)

    for _k in ['v', 'alpha', 'beta']:
        lp_model.remove(lp_vars[_k])
        lp_vars[_k] = gp.tupledict(socp_sol[_k])

    if slack_type == "T":
        lp_cs = add_tc_constraints(
            lp_model, tcfg, vao_sol, lp_vars, S=S, ds=ds,
            is_uphill_dir=is_uphill_dir,
            is_time_constrained=False,
            is_soc_included=False,
        )

        lp_T_slack = lp_model.addVar(lb=0, name="T_slack")
        alpha_fixed = lp_vars["alpha"]
        time_lhs = sum(alpha_fixed[s] for s in range(1, S+1)) * ds
        lp_cs["time"] = gp.tupledict()
        lp_cs["time"][None] = lp_model.addConstr(
            time_lhs - lp_T_slack <= tcfg.T,
            name="time_with_slack"
        )

        # Small U_slack to ensure LP feasibility
        lp_U_slack_plus = lp_model.addVars(range(1, S+1), lb=0, ub=MIN_U_EPS, name="U_slack_plus")
        lp_U_slack_minus = lp_model.addVars(range(1, S+1), lb=0, ub=MIN_U_EPS, name="U_slack_minus")
        lp_model.remove(lp_cs["newton"])
        _c = tcfg.mass * (1 + tcfg.rho) / FORCE_SCALE
        lp_cs["newton"] = lp_model.addConstrs(
            (lp_vars["a"][s] * _c - (
                lp_vars["uPlus"][s] + lp_U_slack_plus[s] - lp_U_slack_minus[s]
                - lp_vars["uMinus"][s]
                - lp_vars["w0"][s] - lp_vars["wtn"][s] - lp_vars["wi"][s]
            ) == 0 for s in range(1, S+1)),
            name="newton_with_slack"
        )

        lp_obj_no_penal = add_eetc_objective(lp_model, tcfg, lp_vars, S=S, ds=ds, is_reg_included=False, scale=ti_scale)
        lp_model.setObjective(
            lp_model.getObjective()
            + LARGE_PENALTY * lp_T_slack
            + LARGE_PENALTY * gp.quicksum(lp_U_slack_plus[s] + lp_U_slack_minus[s] for s in range(1, S+1)),
            gp.GRB.MINIMIZE
        )
    else:
        lp_cs = add_tc_constraints(
            lp_model, tcfg, vao_sol, lp_vars, S=S, ds=ds,
            is_uphill_dir=is_uphill_dir,
            is_time_constrained=False,
            is_soc_included=False,
        )

        lp_U_slack_plus = lp_model.addVars(range(1, S+1), lb=0, name="U_slack_plus")
        lp_U_slack_minus = lp_model.addVars(range(1, S+1), lb=0, name="U_slack_minus")
        lp_model.remove(lp_cs["newton"])
        _c = tcfg.mass * (1 + tcfg.rho) / FORCE_SCALE
        lp_cs["newton"] = lp_model.addConstrs(
            (lp_vars["a"][s] * _c - (
                lp_vars["uPlus"][s] + lp_U_slack_plus[s] - lp_U_slack_minus[s]
                - lp_vars["uMinus"][s]
                - lp_vars["w0"][s] - lp_vars["wtn"][s] - lp_vars["wi"][s]
            ) == 0 for s in range(1, S+1)),
            name="newton_with_slack"
        )

        lp_obj_no_penal = add_eetc_objective(lp_model, tcfg, lp_vars, S=S, ds=ds, is_reg_included=False, scale=ti_scale)
        lp_model.setObjective(
            lp_model.getObjective()
            + LARGE_PENALTY * gp.quicksum(lp_U_slack_plus[s] + lp_U_slack_minus[s] for s in range(1, S+1)),
            gp.GRB.MINIMIZE
        )

    lp_model.update()
    lp_model.Params.OutputFlag = 0
    lp_model.optimize()

    if lp_model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        log_method(
            f"{log_prefix}  Slack fixed-LP {STATUS_NAMES.get(lp_model.Status, lp_model.Status)}!\t\t\t<== WARNING!"
        )
        if lp_model.Status == gp.GRB.INFEASIBLE:
            lp_model.computeIIS()
            pat = run_dir / f"tc_slack_fixed_CB{n_callback}_TC{ti_idx}.ilp"
            pat.parent.mkdir(parents=True, exist_ok=True)
            lp_model.write(str(pat))
        return None, "slack_failed"

    slack_obj = lp_model.ObjVal
    slack_obj_no_penal = lp_obj_no_penal.getValue()
    log_method(f"{log_prefix}  Slack model solved, obj_no_penal={slack_obj_no_penal:.6e}, obj_penal={slack_obj:.6e}")

    cut_expr = compute_optim_cut(
        sp_cs=lp_cs,
        sub_obj_val=slack_obj_no_penal,  # no penalty
        mp_vars=mp_vars,
        mp_vars_bar=vao_sol,
        tcfg=tcfg,
        gcfg=gcfg,
        is_wtn_included=True,
        is_uphill_dir=is_uphill_dir,
    )

    if cut_expr is not None:
        log_method(f"{log_prefix}  Slack fixed-LP optim_cut extracted with obj_no_penal={slack_obj_no_penal:.6e}.")
        return cut_expr, f"slack_{slack_type}_fixed"

    log_method(f"{log_prefix}  Slack fixed-LP dual extraction also failed!\t\t\t<== WARNING!")
    return None, "slack_failed"


def try_extract_optim_cut(
    tc_model: gp.Model,
    mp_vars: dict[str, gp.tupledict],
    vao_sol: dict[str, dict[Any, float]],
    tcfg: TrainConfig,
    ti_idx: int,
    ti_scale: float,
    gcfg: GroundConfigM2,
    is_uphill_dir: bool,
    run_dir: Path,
    n_callback:int,
    log_method: Optional[Callable] = None,
    log_prefix: str = "",
) -> tuple[Optional[gp.LinExpr], str]:
    """Try to extract optimality cut from TC subproblem.

    :return: (cut_expr, model_str)
    """
    log_method = print if log_method is None else log_method

    if tc_model.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        log_method(
            f"{log_prefix} SOCP "
            f"{STATUS_NAMES.get(tc_model.Status, tc_model.Status)} "
            f"(unexpected!)"
        )

        return _try_recover_infeasible_tc(
            tcfg=tcfg,
            gcfg=gcfg,
            vao_sol=vao_sol,
            mp_vars=mp_vars,
            is_uphill_dir=is_uphill_dir,
            log_method=log_method,
            log_prefix=log_prefix,
            ti_scale=ti_scale,
            n_callback=n_callback,
            ti_idx=ti_idx,
        )

    cut_expr = compute_optim_cut(
        sp_cs=tc_model._constraints, sub_obj_val=tc_model.ObjVal,
        mp_vars=mp_vars, mp_vars_bar=vao_sol,
        tcfg=tcfg, gcfg=gcfg,
        is_wtn_included=True, is_uphill_dir=is_uphill_dir,
    )
    if cut_expr is not None:
        log_method(f"{log_prefix}  SOCP optim_cut extracted with obj={tc_model.ObjVal:.6e}.")
        return cut_expr, "socp"

    # SOCP dual extraction failed: fix velocities to reduce to LP, then extract dual
    model_str= "active" if USE_ACTIVE_CONSTRAINTS_LP else "fixed"
    log_method(f"{log_prefix}  DUAL EXTRACTION FAILED, try {model_str}-LP now.")
    socp_sol = extract_gp_solution(tc_model._variables)
    if USE_ACTIVE_CONSTRAINTS_LP:
        lp_m, lp_v, lp_c, active_logs = build_active_constraints_lp(
            tcfg, socp_sol,
            S=gcfg.S, ds=gcfg.ds,
            is_uphill_dir=is_uphill_dir, vao_vars=vao_sol,
            is_reg_included=False,
            obj_scale=ti_scale,
        )
        for msg in active_logs:
            log_method(f"{log_prefix}  {msg}")
    else:
        lp_m, lp_v, lp_c = build_tc_fixed_model(
            tcfg, socp_sol,
            S=gcfg.S, ds=gcfg.ds,
            is_uphill_dir=is_uphill_dir, vao_vars=vao_sol,
            is_reg_included=False,
            obj_scale=ti_scale,
        )
    lp_m.Params.OutputFlag = 0
    lp_m.optimize()

    if lp_m.Status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
        log_method(
            f"{log_prefix}  "
            f"{model_str}-LP {STATUS_NAMES.get(lp_m.Status, lp_m.Status)} "
            f"(unexpected!)\t\t\t<===WARNING!"
        )
        if lp_m.Status == gp.GRB.INFEASIBLE:
            lp_m.computeIIS()
            pat = run_dir / f"tc_{model_str}_CB{n_callback:03d}_TC{ti_idx:02d}.ilp"
            lp_m.write(str(pat))
            log_method(f"{log_prefix}  IIS written to {pat}")
        return None, model_str

    cut_expr = compute_optim_cut(
        sp_cs=lp_c, sub_obj_val=lp_m.ObjVal,
        mp_vars=mp_vars, mp_vars_bar=vao_sol,
        tcfg=tcfg, gcfg=gcfg,
        is_wtn_included=True, is_uphill_dir=is_uphill_dir,
    )
    if cut_expr is None:
        log_method(f"{log_prefix}  LP optim_cut extraction failed."
                   f"(unexpected!)\t\t\t<===WARNING!")
        return None, model_str

    log_method(f"{log_prefix}  {model_str}-LP optim_cut extracted with obj={lp_m.ObjVal:.6e}. (SOCP obj={tc_model.ObjVal:.6e})")
    return cut_expr, model_str


if __name__ == "__main__":
    from models.vao2 import generate_flat_track, build_vao_model
    from utils.gurobi import set_warm_start_vars, extract_gp_solution, set_params_for_robust_solve_socp

    # ====================
    # try larger num_eval_points to see if dual extraction works
    gcfg = GroundConfigM2(num_eval_points=150, ds=50, smooth_scale=0.8)
    vao_flat_sol = generate_flat_track(gcfg)
    vao, vao_vars, vao_cs = build_vao_model(cfg=gcfg, scale=1.0)
    set_warm_start_vars(vao_vars, vao_flat_sol, set_continuous=True)
    vao.Params.OutputFlag = 0
    vao.Params.MIPGap = 0.9
    vao.Params.TimeLimit = 10
    vao.optimize()
    if vao.SolCount == 0:
        raise ValueError(f"Model status is {vao.status} {STATUS_NAMES[vao.status]}, Found no optimal solution.")
    vao_vars_vals = extract_gp_solution(vao_vars)

    tcfg = TrainConfig(train_type="NL_intercity_VIRM6_")
    T_simulated = simulate_sotc(tcfg, vao_vars_vals, S=gcfg.S, ds=gcfg.ds, is_uphill_dir=True)
    T_max = T_simulated * 1.3

    tcfg.set_travel_time(custom_time=T_max, total_distance=gcfg.S * gcfg.ds, verbose=True)

    tc, tc_vars, tc_cs = build_tc_model(cfg=tcfg, S=gcfg.S, ds=gcfg.ds, is_uphill_dir=True, vao_vars=vao_vars_vals)
    set_params_for_robust_solve_socp(tc)
    tc.Params.InfUnbdInfo = 1
    tc.optimize()

    duals = extract_dual_info(tc_cs, tc.Status, verbose=True)
    if duals is not None:
        print(duals.keys())
