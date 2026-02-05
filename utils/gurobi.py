# utils/gurobi.py
"""Gurobi optimization helper functions."""

from typing import Any, Optional
import gurobipy as gp

# Gurobi status codes to human-readable names
STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
}


def extract_gp_solution(gp_vars: dict[str, gp.tupledict | gp.Var] | dict[str, gp.tupledict]) -> dict:
    sol: dict[str, dict[Any, float] | float] = {}
    for var_name, var_dict in gp_vars.items():
        if isinstance(var_dict, dict):
            sol[var_name] = {k: float(v.X) if isinstance(v, gp.Var) else v for k, v in var_dict.items()}
        elif isinstance(var_dict, gp.Var):
            sol[var_name] = float(var_dict.X)
    return sol

def extract_gp_cb_solution(gp_vars: dict[str, gp.tupledict | gp.Var] | dict[str, gp.tupledict], model: gp.Model, binary_vars: list[str] = ["z1", "z2", "z3"]) -> dict:
    sol: dict[str, dict[Any, float] | float] = {}
    for var_name, var_dict in gp_vars.items():
        if isinstance(var_dict, dict):
            if var_name in binary_vars:
                sol[var_name] = {k: round(model.cbGetSolution(v)) for k, v in var_dict.items()}
            else:
                sol[var_name] = {k: model.cbGetSolution(v) for k, v in var_dict.items()}
        elif isinstance(var_dict, gp.Var):
            if var_name in binary_vars:
                sol[var_name] = round(model.cbGetSolution(var_dict))
    return sol

def extract_dual_info(cs: dict[str, gp.tupledict], model_status: int, verbose:bool=False) -> dict[str, dict[Any, float]] | None:
    dual_info: dict[str, dict[Any, float]] = {}
    err_stats: dict[str, int] = {}
    total_err_count = 0
    qc_err_count = 0
    
    if model_status == gp.GRB.OPTIMAL:
        linear_attr = "Pi"
        qc_attr = "QCPi"
    elif model_status == gp.GRB.INFEASIBLE:
        linear_attr = "FarkasDual"
        qc_attr = None  
    else:
        print(f"Warning: Cannot extract duals for status {model_status}")
        return {}
    
    for cs_name, cs_dict in cs.items():
        dual_info[cs_name] = {}
        err_count = 0
        for s in cs_dict.keys():
            constr = cs_dict[s]
            try:
                if isinstance(constr, gp.Constr):
                    dual_info[cs_name][s] = constr.getAttr(linear_attr)
                elif isinstance(constr, gp.QConstr):
                    if qc_attr is not None:
                        dual_info[cs_name][s] = constr.getAttr(qc_attr)
                    else:
                        dual_info[cs_name][s] = 0.0  # 不可行时 QConstr 无对偶
                else:
                    dual_info[cs_name][s] = 0.0
            except (gp.GurobiError, AttributeError) as e:
                err_count += 1
                if isinstance(constr, gp.QConstr):
                    qc_err_count += 1
                dual_info[cs_name][s] = 0.0
                
        if err_count > 0:
            err_stats[cs_name] = err_count
            total_err_count += err_count
    
    if err_stats and total_err_count > qc_err_count:
        if verbose:
            print(f"⚠️ [NoDual]: {total_err_count} | ", end="")
            for cs_name, count in err_stats.items():
                print(f"{cs_name}-{count} | ", end="")
            print()
        return None
    return dual_info


def set_binary_start(v: gp.Var, val: float, debug_mode: bool = False, model: Optional[gp.Model] = None) -> Optional[gp.Constr]:
    constr = None
    if debug_mode:
        if model is None:
            v.LB = 1 if val > 0.5 else 0
            v.UB = 0 if val < 0.5 else 1
        else:
            constr = model.addConstr(v == val)
    v.Start = 0 if val < 0.5 else 1
    return constr


def set_warm_start(
    m: gp.Model, 
    ws_sol: dict[str, dict[Any, float]], 
    debug_on: bool = False,
    set_continuous: bool = False,
    use_eq_constr: bool = False,
) -> Optional[dict[str, dict[Any, gp.Constr]]]:
    constr_dict: dict[str, dict[Any, gp.Constr]] = {}
    for var_name, solution in ws_sol.items():
        for k, val in solution.items():
            full_var_name = f"{var_name}[{k}]"
            var = m.getVarByName(full_var_name)
            if var is not None:
                if var.VType == gp.GRB.BINARY:
                    constr = set_binary_start(var, val, debug_mode=debug_on, model=m if use_eq_constr else None)
                    if constr is not None:
                        constr_dict.setdefault(var_name, {})[k] = constr
                elif set_continuous and var.VType == gp.GRB.CONTINUOUS:
                    var.Start = val
    m.update()
    return constr_dict if use_eq_constr else None


def set_warm_start_vars(
    variables: dict[str, dict[Any, gp.Var]] | dict[str, gp.tupledict],
    ws_sol: dict[str, dict[Any, float]],
    debug_on: bool = False,
    set_continuous: bool = False,
    use_model_eq_constr: bool = False,
    model: Optional[gp.Model] = None,
    suppress_warning: bool = False,
) -> Optional[dict[str, dict[Any, gp.Constr]]]:
    constr_dict: dict[str, dict[Any, gp.Constr]] = {}
    if use_model_eq_constr:
        assert model is not None, "Model must be provided when use_model_eq_constr is True"
    for var_name, var_dict in variables.items():
        solutions = ws_sol.get(var_name, {})
        for key, val in solutions.items():
            if key not in var_dict:
                continue
            if var_dict[key].VType == gp.GRB.BINARY:
                constr = set_binary_start(var_dict[key], val, debug_mode=debug_on, model=model if use_model_eq_constr else None)
                if constr is not None:
                    constr_dict.setdefault(var_name, {})[key] = constr
            elif set_continuous and var_dict[key].VType == gp.GRB.CONTINUOUS:
                var_dict[key].Start = val
                if debug_on and use_model_eq_constr:
                    assert model is not None
                    constr_dict.setdefault(var_name, {})[key] = model.addConstr(var_dict[key] == val)
            else:
                if not suppress_warning:
                    print(f"Warning: Variable {var_name}[{key}] with type {var_dict[key].VType} "
                            f"has no warm start value.")
    return constr_dict if use_model_eq_constr else None


def set_params_for_robust_solve_socp(model: gp.Model):
    """Configure Gurobi parameters for accurate SOCP solving with reliable dual information."""
    model.Params.NumericFocus = 2       
    model.Params.Method = 2             
    model.Params.Presolve = 2           
    model.Params.DualReductions = 0     
    model.Params.Crossover = 2          
    model.Params.ScaleFlag = 2          
    model.Params.BarHomogeneous = 1     
    model.Params.QCPDual = 1            
    model.Params.FeasibilityTol = 1e-6  
    model.Params.OptimalityTol = 1e-6   
    model.Params.BarConvTol = 1e-6      
    model.Params.BarQCPConvTol = 1e-5   
    return


def get_model_details_txt(model: gp.Model) -> str:
    txt = ""
    txt += '>>' * 20 + ' DETAIL RESULT ' + '<<' * 20 + '\n'
    txt += '>>' * 10 + ' VARIABLES ' + '<<' * 10 + '\n'
    
    for var in model.getVars():
        txt += (f"{var.VarName}, VType: {var.VType}, LB: {var.LB}, UB: "
                f"{var.UB}, ObjCoefficient: {var.Obj}, value: {var.X}\n")
    
    txt += ">>" * 10 + ' CONSTRAINTS ' + '<<' * 10 + '\n'
    for constr in model.getConstrs():
        expr = model.getRow(constr)
        lhs_value = 0
        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)
            lhs_value += var.X * coeff
        txt += (f"{constr.ConstrName} with SLACK {constr.Slack:.4f}: "
                f"{model.getRow(constr)} = {lhs_value} ||{constr.Sense}=|| {constr.RHS}\n")
    
    txt += '>>' * 20 + ' DETAIL RESULT ' + '<<' * 20 + '\n'
    return txt


def check_model_constr_vio(model: gp.Model) -> bool:
    vio_constrs = []
    for constr in model.getConstrs():
        if constr.Slack < 0 and abs(constr.Slack) > model.Params.BarConvTol:
            print(constr.ConstrName, constr.RHS, constr.Sense, constr.Slack)
            vio_constrs.append(constr)
    for qconstr in model.getQConstrs():
        if qconstr.QCSlack < 0 and abs(qconstr.QCSlack) > model.Params.BarConvTol:
            print(qconstr.QCName, qconstr.QCRHS, qconstr.QCSense, qconstr.QCSlack)
            vio_constrs.append(qconstr)
    
    if vio_constrs:
        return True
    else:
        return False