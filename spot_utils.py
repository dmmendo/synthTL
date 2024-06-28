import spot
import subprocess
import json

global_lcc = spot.language_containment_checker()

def construct_str(prop_list):
    if len(prop_list) == 0:
        return "1"
    res_str = ""
    if len(prop_list) > 0:
        res_str += "(" + prop_list[0] + ") "
    for i in range(1,len(prop_list)):
        res_str +=  "&& " + i*"X" +"(" + prop_list[i] + ") "    
    return "("+res_str+")"

def trace_to_formula(trace):
    automaton = trace.as_automaton()
    acc_run = automaton.accepting_run()
    #WARNING: if there's no prefix, then the produced formula does not contain the trace, but the trace contains the formula
    prefix_list = []
    for i in range(len(acc_run.prefix)):
        prefix_list.append(spot.bdd_format_formula(automaton.get_dict(), acc_run.prefix[i].label))
    cycle_list = []
    for i in range(len(acc_run.cycle)):
        cycle_list.append(spot.bdd_format_formula(automaton.get_dict(), acc_run.cycle[i].label))
    
    prefix_str = construct_str(prefix_list)
    
    cycle_str = construct_str(cycle_list)
    cycle_condition_str = "G" + "(" + cycle_str + " <-> "+ len(cycle_list)*"X" + cycle_str + ")"
    full_form = prefix_str + " && " + len(prefix_list)*"X" + cycle_str + " && " + len(prefix_list)*"X" + cycle_condition_str
    assert spot.are_equivalent(spot.formula(full_form),automaton)
    return full_form

def get_variables(formula_str):
    spot_formula = spot.formula(formula_str)
    ap = spot.atomic_prop_collect(spot_formula)
    return [str(entry) for entry in ap]

def get_independent_conjucts(formula_str,depth):
    clause_list = get_conjucts(formula_str,depth=depth)
    dependency_map = {}
    for i in range(len(clause_list)):
        var_list = get_variables(clause_list[i])
        if len(var_list) > 0:
            for var in var_list:
                if var not in dependency_map:
                    dependency_map[var] = set()
                dependency_map[var].add(i)
        else:
            if "None" not in dependency_map:
                dependency_map["None"] = set()
            dependency_map["None"].add(i)
    cur_sets = []
    for clause_idx_set in dependency_map.values():
        found_intersection = False
        for ex_set in cur_sets:
            if len(ex_set.intersection(clause_idx_set)) > 0:
                found_intersection = True
                ex_set.update(clause_idx_set)
                break
        if not found_intersection:
            cur_sets.append(clause_idx_set)
    final_clause_list = []
    for idx_set in cur_sets:
        this_clause_list = [clause_list[idx] for idx in idx_set]
        new_clause = " && ".join(["("+entry+")" for entry in this_clause_list])
        final_clause_list.append(new_clause)
    return final_clause_list

def check_satisfiable(formula_str):
    clause_list = get_independent_conjucts(formula_str,depth=2)
    for clause in clause_list:
        if check_formula_contains_formula("0",clause):
            return False
    return True

def check_wellformed(formula_str):
    res = False
    try:
        spot_formula = spot.formula(formula_str)
        res = True
    except:
        res = False
    return res
    
def check_formula_contains_formula(formula,trace_formula,use_contains_split=False):
    global global_lcc
    if global_lcc is None:
        formula_spot = spot.formula(formula)
        trace_spot = spot.formula(trace_formula)
        spot_holds = formula_spot.contains(trace_spot)
        return spot_holds
    else:        
        if use_contains_split:
            return check_splitformula_contains_formula(formula,trace_formula,global_lcc)
        else:
            formula_spot = spot.formula(formula)
            trace_spot = spot.formula(trace_formula)
            spot_holds = global_lcc.contains(formula_spot,trace_spot)
            return spot_holds

def check_splitformula_contains_formula(formula,trace_formula,lcc):
    conj_list = get_conjucts(formula,depth=2)
    trace_spot = spot.formula(trace_formula)
    if lcc is not None:
        for clause in conj_list:
            if not lcc.contains(spot.formula(clause),trace_spot):
                 return False
        return True
        #return all([lcc.contains(spot.formula(clause),trace_spot) for clause in conj_list])
    else:
        for clause in conj_list:
            if not spot.formula(clause).contains(trace_spot):
                 return False
        return True

def get_black_output(ltl_expression,timeout=60):
    cmd_str = "docker exec -e LD_LIBRARY_PATH=/usr/local/lib black_ubuntu black solve -B cmsat -o json -f "
    cmd_list = cmd_str.split()    
    cmd_list.append(ltl_expression)
    result = subprocess.run(cmd_list, stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    res = result.stdout.decode("utf-8")
    error_str = result.stderr.decode("utf-8")
    assert error_str == "", "black failed while trying formula: " + ltl_expression +" with error msg: "+ error_str
    assert len(res) > 0, "black did not return anything!"
    json_res = json.loads(res)
    return json_res

def check_is_black_sat(ltl_expression,timeout=60):
    spot_f = spot.formula(ltl_expression).unabbreviate("WMRie^").negative_normal_form()
    spot_f_str = spot_f.to_str('spot',parenth=True)
    processed_ltl_str = spot_f_str
    var_list = get_variables(processed_ltl_str)
    for var in var_list:
        processed_ltl_str = processed_ltl_str.replace(var," "+var)
    processed_ltl_str = processed_ltl_str.replace("0","False")
    processed_ltl_str = processed_ltl_str.replace("1","True")
    assert len(get_variables(processed_ltl_str)) == len(get_variables(spot_f_str))
    json_res = get_black_output(processed_ltl_str,timeout=timeout)
    return json_res["result"] == 'SAT'

def black_check_formula_contains_formula(formula,trace_formula):
    check_str = "("+trace_formula+") && !("+formula+")"
    return not check_is_black_sat(check_str)

def check_equivalent(formula,trace_formula,use_contains_split=False):
    global global_lcc
    if global_lcc is not None and not use_contains_split:
        return global_lcc.are_equivalent(spot.formula(formula),spot.formula(trace_formula))
        #return spot.are_equivalent(spot.formula(formula),spot.formula(trace_formula))
    else:
        return check_splitformula_contains_formula(formula,trace_formula,global_lcc) and \
            check_splitformula_contains_formula(trace_formula,formula,global_lcc)

def get_counter_example(formula_a,formula_b,ret_trace_formula=False,simplify=True,debug=True):
    #does formula_a contain formula_b?
    check_str = "("+str(formula_b)+") && !("+str(formula_a)+")"
    is_sat,trace = check_satisfiable(check_str, ret_trace_formula=ret_trace_formula, simplify=simplify)
    a_contains_b = not is_sat
    if debug:
        if ret_trace_formula:
            assert spot.formula(formula_b).contains(spot.formula(trace))
        else:
            assert spot.formula(formula_b).contains(trace.as_automaton())
    return a_contains_b, trace

def distribute_unary_op(in_f,op_fnc,next_nnf_to_cnf):
    child_new_f = next_nnf_to_cnf(in_f[0])
    if child_new_f._is(spot.op_And):
        new_f = spot.formula.And([op_fnc(entry) for entry in child_new_f])
    else:
        new_f = op_fnc(child_new_f)
    return new_f    

def distribute_binary_op(in_f,op_fnc,mode,next_nnf_to_cnf):
    lhs_f = next_nnf_to_cnf(in_f[0])
    rhs_f = next_nnf_to_cnf(in_f[1])
    assert mode in ["left","right"]
    if mode == "left" and lhs_f._is(spot.op_And):
        new_f = spot.formula.And([op_fnc(entry,rhs_f) for entry in lhs_f])
    elif mode == "right" and rhs_f._is(spot.op_And):
        new_f = spot.formula.And([op_fnc(lhs_f,entry) for entry in rhs_f])
    else:
        new_f = op_fnc(lhs_f,rhs_f)
    return new_f

def nnf_to_cnf(in_f,depth=None):
    if depth is None:
        next_depth = None
    else:
        next_depth = depth-1
    next_nnf_to_cnf = lambda x : nnf_to_cnf(x,next_depth)
    if len(in_f) == 0 or depth == 0:
        return in_f
    elif in_f._is(spot.op_Or):
        child_list = [next_nnf_to_cnf(entry) for entry in in_f]
        for i in range(len(child_list)):
            if child_list[i]._is(spot.op_And):
                or_list = [child_list[j] for j in range(len(child_list)) if j != i]
                child_and_f = child_list[i]
                and_list = []
                for j in range(len(child_and_f)):
                    and_list.append(spot.formula.Or(or_list+[child_and_f[j]]))
                new_f = spot.formula.And(and_list)
                return new_f.map(next_nnf_to_cnf)
        return spot.formula.Or(child_list)
    elif in_f._is(spot.op_G):
        assert len(in_f) == 1 #G should be unary
        new_f = distribute_unary_op(in_f,spot.formula.G,next_nnf_to_cnf=next_nnf_to_cnf)
        return new_f
    elif in_f._is(spot.op_X):
        assert len(in_f) == 1 #X should be unary
        new_f = distribute_unary_op(in_f,spot.formula.X,next_nnf_to_cnf=next_nnf_to_cnf)
        return new_f    
    elif in_f._is(spot.op_U):
        assert len(in_f) == 2 #U should be binary
        new_f = distribute_binary_op(in_f,spot.formula.U,mode="left",next_nnf_to_cnf=next_nnf_to_cnf)
        return new_f
    elif in_f._is(spot.op_W):
        assert len(in_f) == 2 #W should be binary
        new_f = distribute_binary_op(in_f,spot.formula.W,mode="left",next_nnf_to_cnf=next_nnf_to_cnf)
        return new_f
    elif in_f._is(spot.op_R):
        assert len(in_f) == 2 #R should be binary
        new_f = distribute_binary_op(in_f,spot.formula.R,mode="right",next_nnf_to_cnf=next_nnf_to_cnf)
        return new_f
    elif in_f._is(spot.op_M):
        assert len(in_f) == 2 #M should be binary
        new_f = distribute_binary_op(in_f,spot.formula.M,mode="right",next_nnf_to_cnf=next_nnf_to_cnf)
        return new_f
    else:
        return in_f.map(next_nnf_to_cnf)

def ltl_to_cnf(in_f,depth=None):
    return nnf_to_cnf(in_f.unabbreviate("ie^").negative_normal_form(),depth=depth)

def split_conjuction(in_f):
    if in_f._is(spot.op_And):
        res = []
        for entry in in_f:
            res += split_conjuction(entry)
        return res
    else:
        return [in_f]

def get_conjucts(formula_str,depth=None):
    spot_formula = spot.formula(formula_str)
    conj_spot_formula = ltl_to_cnf(spot_formula,depth=depth)
    list_of_conj = split_conjuction(conj_spot_formula)
    return ["("+str(entry)+")" for entry in list_of_conj]

def distribute_unary_op_over_or(in_f,op_fnc,next_nnf_to_dnf):
    child_new_f = next_nnf_to_dnf(in_f[0])
    if child_new_f._is(spot.op_Or):
        new_f = spot.formula.Or([op_fnc(entry) for entry in child_new_f])
    else:
        new_f = op_fnc(child_new_f)
    return new_f    

def distribute_binary_op_over_or(in_f,op_fnc,mode,next_nnf_to_dnf):
    lhs_f = next_nnf_to_dnf(in_f[0])
    rhs_f = next_nnf_to_dnf(in_f[1])
    assert mode in ["left","right"]
    if mode == "left" and lhs_f._is(spot.op_Or):
        new_f = spot.formula.Or([op_fnc(entry,rhs_f) for entry in lhs_f])
    elif mode == "right" and rhs_f._is(spot.op_Or):
        new_f = spot.formula.Or([op_fnc(lhs_f,entry) for entry in rhs_f])
    else:
        new_f = op_fnc(lhs_f,rhs_f)
    return new_f

def nnf_to_dnf(in_f,depth=None):
    if depth is None:
        next_depth = None
    else:
        next_depth = depth-1
    next_nnf_to_dnf = lambda x : nnf_to_dnf(x,next_depth)
    if len(in_f) == 0 or depth == 0:
        return in_f
    elif in_f._is(spot.op_And):
        child_list = [next_nnf_to_dnf(entry) for entry in in_f]
        for i in range(len(child_list)):
            if child_list[i]._is(spot.op_Or):
                and_list = [child_list[j] for j in range(len(child_list)) if j != i]
                child_or_f = child_list[i]
                or_list = []
                for j in range(len(child_or_f)):
                    or_list.append(spot.formula.And(and_list+[child_or_f[j]]))
                new_f = spot.formula.Or(or_list)
                return new_f.map(next_nnf_to_dnf)
        return spot.formula.And(child_list)
    elif in_f._is(spot.op_F):
        assert len(in_f) == 1 #F should be unary
        new_f = distribute_unary_op_over_or(in_f,spot.formula.F,next_nnf_to_dnf=next_nnf_to_dnf)
        return new_f
    elif in_f._is(spot.op_X):
        assert len(in_f) == 1 #X should be unary
        new_f = distribute_unary_op_over_or(in_f,spot.formula.X,next_nnf_to_dnf=next_nnf_to_dnf)
        return new_f    
    elif in_f._is(spot.op_U):
        assert len(in_f) == 2 #U should be binary
        new_f = distribute_binary_op_over_or(in_f,spot.formula.U,mode="right",next_nnf_to_dnf=next_nnf_to_dnf)
        return new_f
    elif in_f._is(spot.op_W):
        assert len(in_f) == 2 #W should be binary
        new_f = distribute_binary_op_over_or(in_f,spot.formula.W,mode="right",next_nnf_to_dnf=next_nnf_to_dnf)
        return new_f
    elif in_f._is(spot.op_R):
        assert len(in_f) == 2 #R should be binary
        new_f = distribute_binary_op_over_or(in_f,spot.formula.R,mode="left",next_nnf_to_dnf=next_nnf_to_dnf)
        return new_f
    elif in_f._is(spot.op_M):
        assert len(in_f) == 2 #M should be binary
        new_f = distribute_binary_op_over_or(in_f,spot.formula.M,mode="left",next_nnf_to_dnf=next_nnf_to_dnf)
        return new_f
    else:
        return in_f.map(next_nnf_to_dnf)

def ltl_to_dnf(in_f,depth=None):
    return nnf_to_dnf(in_f.unabbreviate("ie^").negative_normal_form(),depth=depth)

def split_disjunction(in_f):
    if in_f._is(spot.op_Or):
        res = []
        for entry in in_f:
            res += split_disjunction(entry)
        return res
    else:
        return [in_f]

def get_disjuncts(formula_str,depth=None):
    spot_formula = spot.formula(formula_str)
    disj_spot_formula = ltl_to_dnf(spot_formula,depth=depth)
    list_of_disj = split_disjunction(disj_spot_formula)
    return ["("+str(entry)+")" for entry in list_of_disj]
