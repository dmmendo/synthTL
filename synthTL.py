from utils import *
import openai
from tqdm import tqdm
import itertools
from matplotlib import pyplot as plt
import context_retriever
import re
import pandas as pd
import json
import numpy as np
import copy

import spot_utils

def check_translation_parsable(prediction):
    try:
        pred_dict = json.loads(prediction)
    except Exception as e:
        error_msg = "\nThe output must be JSON parsable"
        return f"Output cannot be parsed as JSON! error message: {e}" + error_msg
    
    col_name_list = ["Natural language", "LTL"]
    for col_name in col_name_list:
        error_msg = "\nThe output is not in the correct format."
        error_msg += "\nThe output must include each of the following" + str(col_name_list)
        if col_name not in pred_dict:
            return "Output does not contain the "+col_name+" field." + error_msg
    if not spot_utils.check_wellformed(pred_dict["LTL"]):
        return "The provided LTL translation is not well-formed"
    
    return "" #check passed

def check_translation_semantics(node,prediction):
    assert_text = node.assert_text
    if node.translation_type == "template":
        for dcmp_var,dcmp_node in node.dcmp_dict.items():
            assert_text = assert_text.replace(dcmp_node.assert_text,"_"+dcmp_var+"_")
    
    pred_dict = json.loads(prediction)
    pred_translation = pred_dict["LTL"]
    if not spot_utils.check_wellformed(pred_translation):
        error_msg = "\nAll translations must be well-formed. Either there is missing paranthesis, an invalid operator, or invalid variable name."
        return error_msg

    if spot_utils.check_wellformed(pred_translation):
        if not spot_utils.check_satisfiable(pred_translation):
            error_msg = "\nThe translation is trivially false and cannot be trivially false. Please provide another different translation."
            return error_msg
        if spot_utils.check_formula_contains_formula(pred_translation,"1",use_contains_split=True):
            error_msg = "\nThe translation is trivially true and cannot be trivially true. Please provide another different translation."
            return error_msg
            
        pred_vars = spot_utils.get_variables(pred_translation)
        text_vars = get_valid_variables(assert_text)
        dcmp_vars = ["_"+entry+"_" for entry in node.dcmp_dict.keys()]
        for var in pred_vars:
            
            if var not in text_vars and var not in dcmp_vars:
                error_msg = "\nThe translation must only contain variables the following set of variables: " + str(text_vars+dcmp_vars)
                return "Translation cannot contain the variable " + var + error_msg
            
            
            if var[0] != "_" or var[-1] != "_":
                error_msg = "\nAll variables must start and end with underscores."
                return "Translation cannot contain the variable " + var + error_msg

        for var in dcmp_vars:
            if var not in pred_vars:
                error_msg = "\nThe translation must contain all variables from the following set of variables: " + str(dcmp_vars)
                return "Translation is missing variable " + var + error_msg
        
    return "" #pass

def create_translate_prompt(node):
    f = open("translation_prefix_prompt.txt","r")
    prefix = f.read()
    f.close()
    prefix_prompt = prefix
    prefix_prompt += "\nExample Translations:\n"
    for ex in node.translate_fewshots:
        prefix_prompt += "\n" +  json.dumps(ex) + "\n"
    dcmp_translations = dcmp_to_json(node)
    if len(dcmp_translations) > 0:
        dcmp_txt = "\n\nAccount for the following sub-translations of the natural language phrase to formulate the full translation: "
        for ex in dcmp_translations:
            dcmp_txt += "\n" +  ex + "\n"
    else:
        dcmp_txt = ""
    #query_txt = "\nOnly the following variables which appear in the natural language phrase to translate can appear in the LTL translation: "+str(get_valid_variables(node.assert_text))
    query_txt = "\nThe LTL translation contains a subset of the following variables which appear in the natural language phrase: "+str(get_valid_variables(node.assert_text))
    query_txt += "Accounting for the above examples and sub-translations, provide the LTL formula for the following natural language phrase (the translation must only use LTL operators and the format must be in JSON as shown in previous examples): "
    cur_prompt = prefix_prompt + dcmp_txt + query_txt + node.assert_text
    return cur_prompt

def create_template_translate_prompt(node):
    f = open("translation_prefix_prompt.txt","r")
    prefix = f.read()
    f.close()
    prefix_prompt = prefix
    prefix_prompt += "\nAccount for the following example translations:\n"
    for ex in node.translate_fewshots:
        prefix_prompt += "\n" +  json.dumps(ex) + "\n"

    assert_text = node.assert_text
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        assert_text = assert_text.replace(dcmp_node.assert_text,"_"+dcmp_var+"_")

    #cur_var_list = get_valid_variables(assert_text)+["_"+entry+"_"for entry in node.dcmp_dict.keys()]
    if len(node.dcmp_dict) == 0:   
        context_list = [phrase for phrase in retriever.search(query=assert_text,rerank_top_k=5) if node.assert_text not in phrase]
        retrieval_txt = "\nAccount for the following context if needed: "+"\n".join(context_list)
        cur_var_list = get_valid_variables(assert_text)
    else:
        retrieval_txt = ""
        cur_var_list = get_valid_variables(assert_text) + ["_"+entry+"_"for entry in node.dcmp_dict.keys()]
    query_txt = retrieval_txt
    
    query_txt += "\nThe LTL translation may only contain a subset of the following variables: "+str(cur_var_list)
    query_txt += "\nProvide the LTL formula for the following natural language phrase (the translation must only use LTL operators and the format must be in JSON as shown in previous examples): "
    cur_prompt = prefix_prompt + query_txt + "'"+ assert_text + "'" + "\n"
    return cur_prompt    

def dcmp_to_json(node):
    res_list = []
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        if dcmp_node.translation is not None:
            dcmp_dict = {"Natural language" : dcmp_node.assert_text, "LTL": dcmp_node.translation}
            res_list.append(json.dumps(dcmp_dict))
    return res_list

def check_translation(node,prediction):
    msg = check_translation_parsable(prediction)
    if msg != "":
        return msg
    msg =  check_translation_semantics(node,prediction)
    return msg

def get_variables_from_NL(assert_text):
    res_list = []
    cur_list = assert_text.split('_')
    assert len(cur_list) == 0 or len(cur_list)%2 == 1 #for now, variable names cannot contain underscores
    for i in range(1,len(cur_list),2):
        res_list.append("_"+cur_list[i]+"_")
    return list(set(res_list))

def get_valid_variables(assert_text):
    return spot_utils.get_variables(formula_DUT)    

def translate_LLM(node,t_type='regular',given_prompt=None,max_try=5):
    if given_prompt is not None:
        cur_prompt = given_prompt
    elif t_type == 'regular':
        cur_prompt = create_translate_prompt(node)
    elif t_type == 'template':
        cur_prompt = create_template_translate_prompt(node)
    else:
        assert False, "translate t_type " + t_type + " not found!"
    
    pred,is_valid = get_checked_prediction(cur_prompt,node,check_translation,max_try=max_try,
                                           model="gpt-3.5-turbo-0125"
                                           #model="gpt-4-0125-preview"
                                          )
    #response = get_inference_response(cur_prompt,model="gpt-3.5-turbo-0125")
    #pred = response["choices"][0]["message"]["content"]
    try:
        res =  str(json.loads(pred)["LTL"])
        if res.count("(") > res.count(")"):
            res += (res.count("(")-res.count(")")) * ")"
    except:
        res = None
    return res, cur_prompt, pred

def get_unique_index_list(input_list):
    output = []
    seen = set()
    for i,e in enumerate(input_list):
        if e not in seen:
            output.append(i)
            seen.add(e)
    return output

def construct_retranslation_dict(input_node_list,num_try=3,inner_max_try=5,valid_only=False,assert_valid=False):
    input_node_id_list = [get_abstract_node_id(entry) for entry in input_node_list]
    unique_idx_list = get_unique_index_list(input_node_id_list)
    node_list = [input_node_list[idx] for idx in unique_idx_list]
    node_id_list = [input_node_id_list[idx] for idx in unique_idx_list]
    retranslation_dict = dict((node_id,[]) for node_id in node_id_list)
    assert set(input_node_id_list) == set(node_id_list)
    for i in tqdm(range(len(node_list))):
        node_id = node_id_list[i]
        cur_node = node_list[i]
        assert node_id == get_abstract_node_id(cur_node)
        this_prompt = None
        found_valid = False
        try_count = 0
        while (assert_valid and not found_valid) or try_count < num_try:
        #for try_count in range(num_try):
            node_to_retranslate = copy_graph(cur_node)
            node_to_retranslate.parent = cur_node.parent
            print("calling to translate!")
            time.sleep(0.2)
            cur_prompt, pred = node_to_retranslate.translate(mode='LLM',t_type='template',given_prompt=this_prompt,max_try=inner_max_try)
            #error_msg = check_node(node_to_retranslate,formula_DUT)
            #if error_msg != "":
            #    this_prompt = cur_prompt + "\n" + pred + error_msg
            print("in node check!")
            print(get_node_translation(node_to_retranslate))
            time.sleep(0.2)
            is_valid = check_isolated_node_validity(node_to_retranslate)
            print("finished node check!")
            #if is_valid:
            #    error_msg = "Please provide another plausible translation"
            #    this_prompt = cur_prompt + "\n" + pred + error_msg
            if not valid_only or is_valid:
                found_valid = True
                retranslation_dict[node_id].append(get_node_translation(node_to_retranslate))
            try_count += 1
        if not found_valid:
            print("WARNING: could not find valid translation for node!")
            print(node_id)
            print(get_node_translation(node_to_retranslate))
            test_node = node_to_retranslate
            print("is well formed:",spot_utils.check_wellformed(get_node_translation(test_node)))
            print("no bad symbols:","\"" not in get_node_translation(test_node))
            print("uses all abstract variables:",len([1 for entry in test_node.dcmp_dict.keys() if "_"+entry+"_" in get_node_translation(test_node)]) == len(test_node.dcmp_dict))
            if spot_utils.check_wellformed(get_node_translation(test_node)):
                print("only contains valid variables:",set(spot_utils.get_variables(get_node_translation(test_node))).issubset(set(get_valid_variables(test_node.assert_text)+["_"+entry+"_"for entry in test_node.dcmp_dict.keys()])))
                print("contributing node:",check_node_contributing(test_node))
    return retranslation_dict

#node id types:
#unique - for finding a node in a graph
#literal - for decomposition (assert_text)
#abstract - for translation (assert_text.replace(...,abs_var))

def get_literal_node_id(node):
    return node.assert_text

def get_abstract_node_id(node,ret_var_map=False):
    str_dcmp_dict = dict((k,v.assert_text) for k,v in node.dcmp_dict.items())
    return get_abstract_node_id_from_dcmpdict(node.assert_text,str_dcmp_dict,ret_var_map=ret_var_map)

def remove_duplicate_fromlist_keeporder(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_abstract_node_id_from_dcmpdict(assert_text,dcmp_dict,ret_var_map=False):
    abs_text = assert_text
    for dcmp_var,dcmp_node_text in dcmp_dict.items():
        abs_text = abs_text.replace(dcmp_node_text,"_"+dcmp_var+"-SYMBOL_")
    return get_abstract_node_id_from_abstext(abs_text,ret_var_map=ret_var_map)

def normalize_dcmp_dict(assert_text,dcmp_dict):
    node_id,var_map = get_abstract_node_id_from_dcmpdict(assert_text,dcmp_dict,ret_var_map=True)
    new_dict = {}
    for prev_var,new_var in var_map.items():
        this_prev_var = prev_var[1:-1] #remove underscores
        this_new_var = new_var[1:-1]
        new_dict[this_new_var] = dcmp_dict[this_prev_var]
    return new_dict

def get_abstract_node_id_from_abstext(abs_text,ret_var_map=False,input_mode=False):
    if not input_mode:
        abs_var_list = remove_duplicate_fromlist_keeporder(re.findall('_[a-zA-Z0-9_]*-SYMBOL_',abs_text))
    else:
        abs_var_list = remove_duplicate_fromlist_keeporder(re.findall('_[a-zA-Z0-9_]*SYMBOL_',abs_text))
    new_abs_var_list = ["_SYMBOL"+str(i)+"_" for i in range(len(abs_var_list))]
    node_id = abs_text
    for i in range(len(abs_var_list)):
        new_abs_var = "_SYMBOL"+str(i)+"_"
        node_id = node_id.replace(abs_var_list[i],new_abs_var)
    if not ret_var_map:
        return node_id    
    else:
        var_map = dict((abs_var_list[i].replace("-SYMBOL_","_"),new_abs_var_list[i]) for i in range(len(abs_var_list)))
        return node_id, var_map
        
def get_unique_node_id(node):
    if node.parent is not None:
        return get_unique_node_id(node.parent) + ";" + node.assert_text
    else:
        return node.assert_text

class Node:
    def __init__(self,assert_text,parent=None,translate_fewshots=[],decompose_fewshots=[]):
        self.assert_text = assert_text
        self.translation = None
        self.translation_type = None
        self.template_translation = None
        self.parent = parent
        self.dcmp_dict = {}
        self.translate_fewshots = translate_fewshots
        self.decompose_fewshots = decompose_fewshots
    
    def set_dcmp_dict(self,dcmp_str_dict,force_new=True):
        if not force_new and dcmp_str_dict == dict((k,v.assert_text) for k,v in self.dcmp_dict.items()):
            return
        dcmp_str_dict = normalize_dcmp_dict(self.assert_text,dcmp_str_dict)
        self.dcmp_dict = {}
        for dcmp_var,dcmp_str in dcmp_str_dict.items():
            new_node = Node(assert_text=dcmp_str,
                            parent=self,
                            translate_fewshots=self.translate_fewshots,
                            decompose_fewshots=self.decompose_fewshots
                        )
            self.dcmp_dict[dcmp_var] = new_node

    def decompose(self,mode='LLM',decompose_dict={},**kwargs):
        if mode == 'LLM':
            dcmp_str_dict = decompose_LLM(self,**kwargs)
        elif mode == 'cache':
            dcmp_str_dict = get_decompose_cache(self,decompose_dict=decompose_dict)
        else:
            assert False, "decomposition mode not found! " + mode
        self.set_dcmp_dict(dcmp_str_dict)
        return self

    def translate(self,mode='LLM',t_type='regular',translate_dict={},**kwargs):
        self.translation_type = t_type
        cur_prompt = None
        pred = None
        if mode == 'LLM':
            cur_output,cur_prompt,pred = translate_LLM(self,t_type=t_type,**kwargs)
        elif mode == 'cache':
            cur_output = get_translate_cache(self,t_type=t_type,translate_dict=translate_dict)
        elif mode == 'NoRun' and t_type == 'template':
            cur_output = self.template_translation
        elif mode == 'NoRun' and t_type == 'regular':
            cur_output = self.translation
        else:
            assert False, "translation mode not found! " + mode
        
        if cur_output is None:
            cur_template_translation = None
            cur_translation = None
        elif t_type == 'template':
            cur_template_translation = cur_output
            cur_translation = cur_output
            for dcmp_var,dcmp_node in self.dcmp_dict.items():
                if dcmp_node.translation is not None:
                    cur_translation = cur_translation.replace("_"+dcmp_var+"_","("+dcmp_node.translation+")")
        elif t_type == 'regular':
            cur_template_translation = None
            cur_translation = cur_output
        self.translation = cur_translation
        self.template_translation = cur_template_translation
        return cur_prompt, pred

    def check(self,DUT_formula,ret_trace=False,ret_trace_formula=False):
        if self.translation is None:
            return False, None
        elif not spot_utils.check_wellformed(self.translation):
            return False, None

        cur_conjuct = get_conjucts_for_node(self,debug=False)
        if not spot_utils.check_wellformed(cur_conjuct):
            #print("WARNING: this node is not used by the final translation!")
            return False, None
        
        if spot_utils.check_formula_contains_formula(cur_conjuct,DUT_formula,use_contains_split=True):
            return True, None
        else:
            if ret_trace:
                #print("ATTEMPTING TO GET COUNTER EXAMPLE")
                trace = spot_utils.get_counter_example(DUT_formula,cur_conjuct,ret_trace_formula=ret_trace_formula)
                #print("FINISHED")
                return False, trace
            else:
                return False, None

    def clear(self):
        self.translation = None
        self.translation_type = None
        self.template_translation = None
        self.dcmp_dict = {}

def get_root(node):
    if node.parent is None:
        return node
    else:
        return get_root(node.parent)

def get_node_id(node):
    assert False, "deprecated"
    node_id = node.assert_text
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        node_id = node_id.replace(dcmp_node.assert_text,"_"+dcmp_var+"_")
    return node_id

def get_node_assert_text(node):
    assert False, "deprecated"
    return node.assert_text

def get_node_translation(node):
    if len(node.dcmp_dict) == 0:
        return node.translation
    else:
        return node.template_translation

def find_descendant_by_id(node,targ_node_id,get_node_id_func):
    if get_node_id_func(node) == targ_node_id:
        return node
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        desc_node = find_descendant_by_id(dcmp_node,targ_node_id,get_node_id_func=get_node_id_func)
        if desc_node is not None:
            return desc_node
    return None    

def find_descendant(node,targ_node_id):
    if get_unique_node_id(node) == targ_node_id:
        return node
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        desc_node = find_descendant(dcmp_node,targ_node_id)
        if desc_node is not None:
            return desc_node
    return None

def copy_graph(node):
    new_node = Node(assert_text=node.assert_text)
    new_node.translation = node.translation
    new_node.translation_type = node.translation_type
    new_node.template_translation = node.template_translation
    new_node.translate_fewshots = node.translate_fewshots
    new_node.decompose_fewshots = node.decompose_fewshots
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        new_node.dcmp_dict[dcmp_var] = copy_graph(dcmp_node)
        new_node.dcmp_dict[dcmp_var].parent = new_node
    return new_node

def get_conjucts_for_node(node,verbose=False,ret_list=False,debug=True,omit_trivial=True,depth=None):
    tmp_dcmp_dict = node.dcmp_dict
    tmp_template_translation = node.template_translation
    tmp_translation = node.translation
    cur_root = get_root(node)
    all_translation = cur_root.translation
    
    node_identifier = "specialAP"
    assert node_identifier not in cur_root.translation, cur_root.translation
    node.dcmp_dict = {}
    node.template_translation = None
    node.translation = node_identifier

    dfs_translate(cur_root,mode='NoRun',t_type='template')
    all_conjucts = spot_utils.get_conjucts(cur_root.translation,depth=depth)
    abstract_conjucts_list = [prop for prop in all_conjucts if node_identifier in prop]
    conjucts_for_node = " && ".join(abstract_conjucts_list)
    res = conjucts_for_node.replace(node_identifier,"("+tmp_translation+")")
    #abstract_conjucts_list = [prop.replace(node_identifier,"("+tmp_translation+")") for prop in all_conjucts if node_identifier in prop]
    #res = " && ".join(abstract_conjucts_list)
    node.dcmp_dict = tmp_dcmp_dict
    node.template_translation = tmp_template_translation
    node.translation = tmp_translation
    dfs_translate(cur_root,mode='NoRun',t_type='template')
    assert not debug or spot_utils.check_equivalent(" && ".join(all_conjucts).replace(node_identifier,"("+tmp_translation+")"),all_translation)
    if not ret_list:
        return res
    else:
        conjucts_for_node_list = [clause for clause in spot_utils.get_conjucts(res) if not omit_trivial or not spot_utils.check_equivalent("1",clause)]
        return conjucts_for_node_list

def get_disjuncts_for_node(node,verbose=False,ret_list=False):
    tmp_dcmp_dict = node.dcmp_dict
    tmp_template_translation = node.template_translation
    tmp_translation = node.translation
    cur_root = get_root(node)
    all_translation = cur_root.translation
    
    node_identifier = "specialAP"
    assert node_identifier not in cur_root.translation, cur_root.translation
    node.dcmp_dict = {}
    node.template_translation = None
    node.translation = node_identifier

    dfs_translate(cur_root,mode='NoRun',t_type='template')
    all_disjuncts = spot_utils.get_disjuncts(cur_root.translation)
    abstract_disjuncts_list = [prop for prop in all_disjuncts if node_identifier in prop]
    disjuncts_for_node = " | ".join(abstract_disjuncts_list)
    res = disjuncts_for_node.replace(node_identifier,"("+tmp_translation+")")
    #abstract_disjuncts_list = [prop.replace(node_identifier,"("+tmp_translation+")") for prop in all_disjuncts if node_identifier in prop]
    #res = " && ".join(abstract_disjuncts_list)
    
    node.dcmp_dict = tmp_dcmp_dict
    node.template_translation = tmp_template_translation
    node.translation = tmp_translation
    dfs_translate(cur_root,mode='NoRun',t_type='template')
    try:
        is_equal = spot_utils.check_equivalent(" | ".join(all_disjuncts).replace(node_identifier,"("+tmp_translation+")"),all_translation)
    except:
        is_equal = True
        print("WARNING: exception thrown when verifying get_disjuncts_for_node")
    assert is_equal
    if not ret_list:
        return res
    else:
        if len(abstract_disjuncts_list) > 0:
            disjuncts_for_node_list = [clause for clause in spot_utils.get_disjuncts(res) if not spot_utils.check_equivalent("0",clause)]
            return disjuncts_for_node_list
        else:
            return []

def get_all_ancestors(node,inclusive=True):
    if inclusive:
        cur_list = [node]
    else:
        cur_list = []
    if node.parent is not None:
        cur_list += get_all_ancestors(node.parent,inclusive=True)
    return cur_list

def get_all_descendants(node,inclusive=True):
    if inclusive:
        cur_list = [node]
    else:
        cur_list = []
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        cur_list += get_all_descendants(dcmp_node,inclusive=True)
    return cur_list

def dfs_decompose(node,mode='LLM',max_depth=None,decompose_dict={},**kwargs):
    if max_depth is not None and max_depth == 0:
        return node
    node.decompose(mode,decompose_dict=decompose_dict,**kwargs)
    if max_depth is not None:
        next_max_depth = max_depth-1
    else:
        next_max_depth = None
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        dfs_decompose(dcmp_node,mode=mode,max_depth=next_max_depth,decompose_dict=decompose_dict,**kwargs)
    return node

def dfs_translate(node,mode='LLM',t_type='regular',translate_dict={},**kwargs):
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        dfs_translate(dcmp_node,mode=mode,t_type=t_type,translate_dict=translate_dict,**kwargs)
    if len(node.dcmp_dict.items()) == 0:
        node.translate(mode,t_type='regular',translate_dict=translate_dict,**kwargs)
    else:
        node.translate(mode,t_type=t_type,translate_dict=translate_dict,**kwargs)

def get_translate_cache(node,translate_dict,t_type='regular'):
    if t_type == 'regular':
        assert_text = node.assert_text
    elif t_type == 'template':
        abs_node_id = get_abstract_node_id(node)
        assert_text = abs_node_id
        assert assert_text in translate_dict, assert_text
    else:
        assert False
    if assert_text in translate_dict:
        res = translate_dict[assert_text]
    else:
        print("WARNING:",assert_text,"not in cached translations!")
        res = None
    return res

def get_decompose_cache(node,decompose_dict):
    if node.assert_text in decompose_dict:
        dcmp_dict = decompose_dict[node.assert_text]
        res = dict((k,v) for k,v in dcmp_dict.items() if v != "" and v != node.assert_text)
    else:
        print("WARNING:",node.assert_text,"not in cached decompositions!")
        res = {}
    return res
    
def create_decompose_dict(fname):
    decompose_df = pd.read_excel(fname)
    decompose_dict = {}
    for idx,row in decompose_df.iterrows():
        dcmp_dict = json.loads(row['Decomposition'])
        dcmp_dict = dict((k,v) for k,v in dcmp_dict.items() if v != "" and v != row['Natural language'])
        dcmp_dict = normalize_dcmp_dict(row['Natural language'],dcmp_dict)
        decompose_dict[row['Natural language']] = dcmp_dict
    return decompose_dict

def create_translate_dict(fname):
    translate_df = pd.read_excel(fname)
    translate_dict = {}
    for idx,row in translate_df.iterrows():
        translate_dict[row['Natural language']] = row['LTL']
        if 'Decomposed Natural language' in row and 'Template' in row:
            assert "SYMBOL_" in row['Decomposed Natural language']
            abs_node_id,var_map = get_abstract_node_id_from_abstext(row['Decomposed Natural language'],ret_var_map=True,input_mode=True)
            cur_template = row['Template']
            for prev_var,new_var in var_map.items():
                cur_template = cur_template.replace(prev_var,new_var)
            translate_dict[abs_node_id] = cur_template
            #translate_dict[row['Decomposed Natural language']] = row['Template']
    return translate_dict

def get_max_depth(node):
    cur_max = 0
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        cur_max = max(cur_max,get_max_depth(dcmp_node))
    return 1 + cur_max

def graph_to_str(node,depth=0,formula_DUT=None,cur_var_name=None):
    indent = "".join(["\t" for i in range(depth)])
    #print(indent,"Text:",node.assert_text)
    #print(indent,"Translation:",node.translation)
    cur_str=""
    assert_text = node.assert_text
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        assert_text = assert_text.replace(dcmp_node.assert_text,dcmp_var)
    cur_str += "\n" + indent + " Text: " + assert_text
    if cur_var_name is not None:
        cur_str += "\n" + indent + " Symbol: " + cur_var_name
    if len(node.dcmp_dict) == 0:
        cur_str += "\n" + indent + " Translation: " + str(node.translation)
    else:
        cur_str += "\n" + indent + " Translation: " + str(node.template_translation)
    if formula_DUT is not None:
        DUT_holds,_ = node.check(formula_DUT)
        cur_str += "\n" + indent + " Holds on DUT: " + str(DUT_holds)
        #if not DUT_holds:
        #    cur_str += "\n" + indent + " conjucts: " + str(get_conjucts_for_node(node,verbose=False))
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        cur_str += graph_to_str(dcmp_node,depth+1,formula_DUT=formula_DUT,cur_var_name=dcmp_var)
    return cur_str

def check_is_translation_trivial(cur_graph):
    return spot_utils.check_formula_contains_formula(cur_graph.translation,"1",use_contains_split=True) or \
            spot_utils.check_formula_contains_formula("0",cur_graph.translation)

def get_abstract_atomic_proposition(cur_graph,node_id,node_to_abstract_dict):
    if node_id not in node_to_abstract_dict:
        node_to_abstract_dict[node_id] = "_abstractAP"+str(len(node_to_abstract_dict))+"_"
    return node_to_abstract_dict[node_id]

def symbolic_abstract_check(new_graph,node_id,custom_get_node_id,abstraction_state,check_func):
    node_to_abstract_dict = abstraction_state
    cur_node = find_descendant_by_id(new_graph,node_id,get_node_id_func=custom_get_node_id)
    assert cur_node is not None, node_id
    abstract_AP = get_abstract_atomic_proposition(new_graph,node_id,node_to_abstract_dict)
    assert new_graph.translation is None or abstract_AP not in new_graph.translation, new_graph.translation
    cur_node.dcmp_dict = {}
    cur_node.template_translation = None
    cur_node.translation = abstract_AP
    dfs_translate(new_graph,mode='NoRun',t_type='template')
    #is_fail = check_is_translation_trivial(new_graph)
    is_fail = check_func(new_graph)
    return is_fail, new_graph

def symbolic_abstract_check_translation_trivial(new_graph,node_id,custom_get_node_id,abstraction_state):
    return symbolic_abstract_check(new_graph,node_id,custom_get_node_id,abstraction_state,check_func=check_is_translation_trivial)

def symbolic_abstract_check_all_ancestors_visited(new_graph,node_id,custom_get_node_id,abstraction_state,visited_node_ids):
    if node_id in visited_node_ids:
        cur_func = lambda cur_graph : False
    else:
        cur_func = lambda cur_graph : True
    return symbolic_abstract_check(new_graph,node_id,custom_get_node_id,abstraction_state,check_func=cur_func)
    
def get_min_fail_abstraction(root,abstractize_node_and_check_func,node_list=None,**kwargs):
    custom_get_node_id = get_unique_node_id
    cur_graph = copy_graph(root)
    if node_list is None:
        node_list = get_all_descendants(cur_graph)
    min_abs_list = set([custom_get_node_id(node) for node in node_list])
    assert len(node_list) == len(min_abs_list) #assume for now that node ids are unique
    available_id_list = set([custom_get_node_id(node) for node in node_list if len(node.dcmp_dict) == 0])
    abstraction_state = {}
    while len(available_id_list) > 0:
        cur_node_id = available_id_list.pop()
        new_graph = copy_graph(cur_graph)
        is_fail, new_graph = abstractize_node_and_check_func(
            new_graph=new_graph,
            node_id=cur_node_id,
            custom_get_node_id=custom_get_node_id,
            abstraction_state=abstraction_state,
            **kwargs)
        if is_fail:
            min_abs_list.remove(cur_node_id)
            cur_graph = new_graph
            parent_node = find_descendant_by_id(cur_graph,cur_node_id,get_node_id_func=custom_get_node_id).parent
            if parent_node is not None:
                is_sibling_available = False
                for dcmp_node in get_all_descendants(parent_node):
                    if custom_get_node_id(dcmp_node) in available_id_list:
                        is_sibling_available = True
                        break
                if not is_sibling_available and custom_get_node_id(parent_node) in min_abs_list:
                    available_id_list.add(custom_get_node_id(parent_node))
    return min_abs_list, cur_graph
    
def get_visited_abstraction(cur_graph,visited_node_ids):
    id_list, abs_graph = get_min_fail_abstraction(cur_graph,
                                                  abstractize_node_and_check_func=symbolic_abstract_check_all_ancestors_visited,
                                                  visited_node_ids=visited_node_ids)
    return abs_graph

def get_ablate_formulae_for_node(cur_graph,node,use_copy=True):
    custom_get_node_id = get_unique_node_id
    if use_copy:
        test_graph = copy_graph(cur_graph)
        test_node = find_descendant_by_id(test_graph,custom_get_node_id(node),get_node_id_func=custom_get_node_id)
    else:
        test_graph = cur_graph
        test_node = node
    test_node.dcmp_dict = {}
    test_node.translation = "(1)"
    dfs_translate(test_graph,mode='NoRun',t_type='template')
    true_ablate_formula = test_graph.translation
    
    test_node.translation = "(0)"
    dfs_translate(test_graph,mode='NoRun',t_type='template')
    false_ablate_formula = test_graph.translation
    return true_ablate_formula, false_ablate_formula

def get_possible_culprits(root,formula_DUT):
    is_holds,_ = root.check(formula_DUT,ret_trace=False)
    if is_holds:
        return []
    else:
        res_list = [root]
        for dcmp_node in root.dcmp_dict.values():
            res_list += get_possible_culprits(dcmp_node,formula_DUT)
        return res_list

def get_prune_condition(root,culprits,formula_DUT,fail_nodes=None):
    custom_get_node_id = get_unique_node_id
    if fail_nodes is None:
        fail_nodes = get_possible_culprits(root,formula_DUT)
    hold_nodes = []
    for node in get_all_descendants(root):
        if node not in fail_nodes:
            hold_nodes.append(node)
    special_symbol = "SPECIALSYMBOL"
    assert special_symbol not in root.translation
    prune_culprit = []
    for c_node in culprits:
        found_hold = False
        for h_node in hold_nodes:
            tmp_g = copy_graph(root)
            cur_h_node = find_descendant_by_id(tmp_g,custom_get_node_id(h_node),get_node_id_func=custom_get_node_id)
            cur_h_node.translation = special_symbol
            cur_h_node.template_translation = None
            cur_h_node.dcmp_dict.clear()
            dfs_translate(tmp_g,mode='NoRun',t_type='template')
            cur_c_node = find_descendant_by_id(tmp_g,custom_get_node_id(c_node),get_node_id_func=custom_get_node_id)
            trigger_cond = get_conjucts_for_node(cur_c_node,debug=False,depth=get_max_depth(root))
            if special_symbol in trigger_cond:
                found_hold = True
                break
        prune_culprit.append(found_hold)
    return prune_culprit

def filter_if_dependancy_holds(cur_graph,node_list,formula_DUT,fail_nodes=None):
    prune_cond = get_prune_condition(cur_graph,node_list,formula_DUT,fail_nodes=fail_nodes)
    res = [node_list[i] for i in range(len(node_list)) if prune_cond[i] is False]
    return res

def ablation_abstract_check(new_graph,node_id,custom_get_node_id,abstraction_state,formula_DUT):
    node_to_abstract_dict = abstraction_state
    cur_node = find_descendant_by_id(new_graph,node_id,get_node_id_func=custom_get_node_id)
    assert cur_node is not None, node_id
    abstract_AP = get_abstract_atomic_proposition(new_graph,node_id,node_to_abstract_dict)
    assert abstract_AP not in new_graph.translation, new_graph.translation
    cur_node.dcmp_dict = {}
    cur_node.template_translation = None
    org_translation = new_graph.translation
    
    cur_node.translation = "1"
    dfs_translate(new_graph,mode='NoRun',t_type='template')
    true_ablate_formula = new_graph.translation
    #true_ablate,_ = new_graph.check(formula_DUT,ret_trace=False)
    true_ablate = spot_utils.check_formula_contains_formula(true_ablate_formula,formula_DUT,use_contains_split=True)
    
    cur_node.translation = "0"
    dfs_translate(new_graph,mode='NoRun',t_type='template')
    false_ablate_formula = new_graph.translation
    #false_ablate,_ = new_graph.check(formula_DUT,ret_trace=False)
    false_ablate = spot_utils.check_formula_contains_formula(false_ablate_formula,formula_DUT,use_contains_split=True)

    true_ablate_contains_org = spot_utils.check_formula_contains_formula(true_ablate_formula,org_translation,use_contains_split=True)
    false_ablate_contains_org = spot_utils.check_formula_contains_formula(false_ablate_formula,org_translation,use_contains_split=True)
    #assert true_ablate_contains_org or false_ablate_contains_org
    #assert not (true_ablate_contains_org and false_ablate_contains_org) or spot_utils.check_equivalent(true_ablate_formula,false_ablate_formula) \
    #    or true_ablate or false_ablate
    #assert not (true_ablate and false_ablate) or spot_utils.check_equivalent(true_ablate_formula,false_ablate_formula)
    
    #if cur_node.assert_text == "HTRANS is IDLE":
    #    print(true_ablate,false_ablate)
    #    print(true_ablate_contains_org,false_ablate_contains_org)
    #    print(cur_node.parent.translation)
    if not true_ablate and not false_ablate:
        is_fail = True
    else:
        #print("true:",true_ablate,"false:",false_ablate)
        is_fail = False
    if true_ablate_contains_org and false_ablate_contains_org:
        is_fail = False #cannot prune away this node
        print("cannot prune:",cur_node.assert_text)
        cur_node.translation = "0"
        dfs_translate(new_graph,mode='NoRun',t_type='template') 
    elif true_ablate_contains_org:
        cur_node.translation = "1"
        dfs_translate(new_graph,mode='NoRun',t_type='template')
    elif false_ablate_contains_org:
        cur_node.translation = "0"
        dfs_translate(new_graph,mode='NoRun',t_type='template')
    else:
        is_fail = False #cannot prune away this node
        print("cannot prune:",cur_node.assert_text)
        cur_node.translation = "0"
        dfs_translate(new_graph,mode='NoRun',t_type='template')
    return is_fail, new_graph

def get_independent_conjuct_list(cur_graph,min_abs_list,abs_graph):
    custom_get_node_id = get_unique_node_id
    mutable_list = set()
    for node_id in min_abs_list:
        org_node = find_descendant_by_id(cur_graph,node_id,get_node_id_func=custom_get_node_id)
        abs_node = find_descendant_by_id(abs_graph,node_id,get_node_id_func=custom_get_node_id)
        is_mutable = True
        for dcmp_var,dcmp_node in org_node.dcmp_dict.items():
            if custom_get_node_id(dcmp_node) not in min_abs_list:
                is_mutable = False
                break
        if is_mutable:
            mutable_list.add(node_id)
    return mutable_list

def get_node_list_to_update_with_ablation(cur_graph,formula_DUT,ret_only_independent=False,possible_culprits=None):
    custom_get_node_id = get_unique_node_id
    if possible_culprits is None:
        possible_culprits = get_possible_culprits(cur_graph,formula_DUT)
    min_abs_list, abs_graph = get_min_fail_abstraction(cur_graph,
                                                       abstractize_node_and_check_func=ablation_abstract_check,
                                                       node_list=possible_culprits,
                                                       formula_DUT=formula_DUT)
    if not ret_only_independent:
        node_to_update_list = [find_descendant_by_id(cur_graph,node_id,get_node_id_func=custom_get_node_id) for node_id in min_abs_list]
    else:
        #returns a subset of min_abs_list where the nodes pertain to an independent conjuct, 
        #i.e., if those nodes are re-translated, then the ablated node set is gaurenteed to not change, and its impact is local to the independent subset 
        mutable_list = get_independent_conjuct_list(cur_graph,min_abs_list,abs_graph)
        node_to_update_list = [find_descendant_by_id(cur_graph,node_id,get_node_id_func=custom_get_node_id) for node_id in mutable_list]
    return node_to_update_list

def get_culprit_batch(graph_list,formula_DUT,depth=1):
    translation_list = []
    for g in graph_list:
        for node in get_all_descendants(g):
            cur_conjuct = get_conjucts_for_node(node,debug=False,depth=depth)
            translation_list.append(cur_conjuct)
    hold_translation_set = find_hold_set_rawconjmerge(translation_list,formula_DUT=formula_DUT)
    res_list = []
    i = 0
    for g in graph_list:
        cur_culprit_list = []
        for node in get_all_descendants(g):
            if translation_list[i] not in hold_translation_set:
                cur_culprit_list.append(node)
            i += 1
        res_list.append(cur_culprit_list)
    return res_list

def check_isolated_node_validity(node):
    #returns bool
    if len(node.dcmp_dict) == 0:
        #1. well-formed
        #2. not trivially true
        #3. not trivially false
        cur_translation = node.translation
        return spot_utils.check_wellformed(cur_translation) \
            and "\"" not in cur_translation \
            and not spot_utils.check_formula_contains_formula(cur_translation,"1",use_contains_split=True) \
            and spot_utils.check_satisfiable(cur_translation) \
            and set(spot_utils.get_variables(cur_translation)).issubset(set(get_valid_variables(node.assert_text)))
    else:
        #1. well-formed
        #2. all decomposition are used in template translation
        #3. not trivially true
        #4. not trivially false
        #5. uses variables in the decomposition or target module
        #6. all decompositions are contributing
        cur_translation = node.template_translation
        return spot_utils.check_wellformed(cur_translation) \
            and "\"" not in cur_translation \
            and len([1 for entry in node.dcmp_dict.keys() if "_"+entry+"_" in cur_translation]) == len(node.dcmp_dict) \
            and not spot_utils.check_formula_contains_formula(cur_translation,"1",use_contains_split=True) \
            and spot_utils.check_satisfiable(cur_translation) \
            and set(spot_utils.get_variables(cur_translation)).issubset(set(get_valid_variables(node.assert_text)+["_"+entry+"_"for entry in node.dcmp_dict.keys()])) \
            and check_node_contributing(node)

def check_node_contributing(cur_node):
    #assume translation is non-trivial
    #if noncontributing, then setting node 0 or 1 results in the same formula
    test_node = copy_graph(cur_node)
    for dcmp_var,dcmp_node in test_node.dcmp_dict.items():
        dcmp_node.set_dcmp_dict({})
        dcmp_node.translation = "_"+dcmp_var+"_"
    dfs_translate(test_node,mode='NoRun',t_type='template')
    visited_set = set()
    for i in range(len(cur_node.dcmp_dict.values())):
        cur_test_node = copy_graph(test_node)
        test_dcmp_node = list(cur_test_node.dcmp_dict.values())[i]
        assert test_dcmp_node.assert_text not in visited_set
        visited_set.add(test_dcmp_node.assert_text)
        true_ablate_formula, false_ablate_formula = get_ablate_formulae_for_node(cur_test_node,test_dcmp_node,use_copy=False)
        #if spot_utils.check_equivalent(true_ablate_formula, false_ablate_formula):
        if spot_utils.check_formula_contains_formula(true_ablate_formula, false_ablate_formula,use_contains_split=True) \
            and spot_utils.check_formula_contains_formula(false_ablate_formula,true_ablate_formula,use_contains_split=True): 
            return False
    return True

def get_breadthfirst_nodecheck(cur_graph):
    #assume translation is non-trivial
    #if noncontributing, then setting node 0 or 1 results in the same formula
    if not check_isolated_node_validity(cur_graph):
        return [cur_graph]
    else:
        for dcmp_var, dcmp_node in cur_graph.dcmp_dict.items():
            child_list = get_breadthfirst_nodecheck(dcmp_node)
            if len(child_list) > 0:
                return child_list
        return []

def check_graph_validity(cur_graph):
    #returns empty list if graph is valid (i.e., a non=trivial and well-formed translation)
    #invalid_list = [node for node in get_all_descendants(cur_graph) if not check_isolated_node_validity(node)]
    invalid_list = get_breadthfirst_nodecheck(cur_graph)
    #print("isolated check:",len(invalid_list))
    if len(invalid_list) > 0:
        return invalid_list
    
    invalid_list = get_depthfirst_trivial_translation(cur_graph)
    #print("nontrivial check:",len(invalid_list))
    return invalid_list
        
    #invalid_list = get_breadthfirst_noncontributing_nodes(cur_graph,cur_node=cur_graph)
    #print("noncontributing check:",len(invalid_list))
    #return invalid_list 
    
def get_depthfirst_trivial_translation(cur_graph):
    assert not None in [node.translation for node in get_all_descendants(cur_graph)]
    #returns a list, if empty, then graph has no trivial conjucts
    for dcmp_var, dcmp_node in cur_graph.dcmp_dict.items():
        child_list = get_depthfirst_trivial_translation(dcmp_node)
        if len(child_list) > 0:
            return child_list
    if check_is_translation_trivial(cur_graph):
        tmp_parent = cur_graph.parent
        cur_graph.parent = None #for unique_node_id to work
        custom_get_node_id = get_unique_node_id
        min_abs_list, abs_graph = get_min_fail_abstraction(cur_graph,abstractize_node_and_check_func=symbolic_abstract_check_translation_trivial)
        res_list = [find_descendant_by_id(cur_graph,node_id,get_node_id_func=custom_get_node_id) for node_id in min_abs_list]
        assert not None in res_list, res_list
        cur_graph.parent = tmp_parent
        return res_list
    else:
        return []

def get_conjuct_clause_dependency(cur_graph,get_clauses_for_node_func=get_conjucts_for_node):
    #returns dict of clause to list of nodes which contribute to the clause
    #all conjucts of the root are included
    nontrivial_list = get_clauses_for_node_func(cur_graph,ret_list=True)
    clause_to_node_dict = dict((clause,[]) for clause in nontrivial_list)
    
    for node in get_all_descendants(cur_graph):
        for node_clause in get_clauses_for_node_func(node,ret_list=True):
            for targ_clause in clause_to_node_dict:
                if node not in clause_to_node_dict[targ_clause] and spot_utils.check_equivalent(node_clause,targ_clause):
                    clause_to_node_dict[targ_clause].append(node)

    return clause_to_node_dict

def get_counterex_for_node(node,formula_DUT,mode="1-1"):
    assert mode in ["random","1-1","all"]
    tmp_list = [get_conjucts_for_node(sub_node,ret_list=False) for sub_node in get_all_descendants(node)]
    conjuct_list = []
    for clause in tmp_list:
        is_duplicate = False
        for entry in conjuct_list:
            if spot_utils.check_equivalent(clause,entry):
                is_duplicate = True
                break
        if not is_duplicate:
            conjuct_list.append(clause)
    conjuct_list = [clause for clause in conjuct_list if not spot_utils.check_formula_contains_formula(clause,formula_DUT)]
    
    counter_ex_list = []
    if mode == "1-1":
        for i in range(len(conjuct_list)):
            assert not spot_utils.check_formula_contains_formula(conjuct_list[i],formula_DUT)
            is_hold, counter_ex = spot_utils.get_counter_example(conjuct_list[i],formula_DUT,ret_trace_formula=True)
            assert not is_hold
            assert spot_utils.check_formula_contains_formula(formula_DUT,counter_ex), spot_utils.check_formula_contains_formula(conjuct_list[i],counter_ex)
            is_duplicate = False
            for entry in counter_ex_list:
                if spot_utils.check_equivalent(entry,counter_ex):
                    is_duplicate = True
                    break
            if not is_duplicate:
                counter_ex_list.append(counter_ex)
    elif mode == "all":
        unique_conjuct_list = []
        for i in range(len(conjuct_list)):
            is_hold, counter_ex = spot_utils.get_counter_example(conjuct_list[i],formula_DUT,ret_trace_formula=True)
            assert not is_hold
            assert spot_utils.check_formula_contains_formula(formula_DUT,counter_ex)
            is_duplicate = False
            for entry in counter_ex_list:
                if spot_utils.check_equivalent(entry,counter_ex):
                    is_duplicate = True
                    break
            if not is_duplicate:
                counter_ex_list.append(counter_ex)
                unique_conjuct_list.append(conjuct_list[i])
        for i in range(1,len(unique_conjuct_list)):
            for comb in itertools.combinations(unique_conjuct_list,i+1):
                cur_formula = "&&".join(["("+clause+")" for clause in comb])
                is_hold, counter_ex = spot_utils.get_counter_example(cur_formula,formula_DUT,ret_trace_formula=True)
                assert not is_hold 
                assert spot_utils.check_formula_contains_formula(formula_DUT,counter_ex)
                is_duplicate = False
                for entry in counter_ex_list:
                    if spot_utils.check_equivalent(entry,counter_ex):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    counter_ex_list.append(counter_ex)
    elif mode == "random":
        select_idx = np.random.randint(len(conjuct_list))
        assert not spot_utils.check_formula_contains_formula(conjuct_list[select_idx],formula_DUT)
        is_hold, counter_ex = spot_utils.get_counter_example(conjuct_list[select_idx],formula_DUT,ret_trace_formula=True)
        assert not is_hold 
        assert spot_utils.check_formula_contains_formula(formula_DUT,counter_ex), counter_ex
        counter_ex_list.append(counter_ex)

    return counter_ex_list

def get_possible_decomposition_for_node(node,redecomposition_dict):
    literal_node_id = get_literal_node_id(node)
    dcmp_id_list = [get_abstract_node_id_from_dcmpdict(literal_node_id,dcmp_dict) for dcmp_dict in redecomposition_dict[literal_node_id]]
    unique_idx_list = get_unique_index_list(dcmp_id_list)
    return [redecomposition_dict[literal_node_id][idx] for idx in unique_idx_list]

def flatten_list(xss):
    return [x for xs in xss for x in xs]

def calc_decompose_coverage(assert_text,str_dcmp_dict):
    #char-based coverage
    cur_assert_text = assert_text
    total_coverage = 0
    for dcmp_txt in str_dcmp_dict.values():
        prev_len = len(cur_assert_text)
        cur_assert_text = cur_assert_text.replace(dcmp_txt,"")
        after_len = len(cur_assert_text)
        cur_coverage = prev_len - after_len
        total_coverage += cur_coverage
    return total_coverage / len(assert_text)

def get_deconstrain_decomposition_for_node(cur_graph,node,retranslation_dict,redecomposition_dict,
                                         get_node_hash=get_unique_node_id,
                                         check_trivial=False,
                                         filter_res=False,
                                         skip_set=set(),
                                         cur_cache={},
                                         visited_set=set(),
                                         mode="prune"):
    assert mode in ["prune","all","one_prune"]
    #visited_set.add(get_node_hash(node))    
    if get_node_hash(node) in skip_set:
        visited_set.add(get_node_hash(node))
        if len(node.dcmp_dict) == 0:
            res_list = [copy_graph(node)]
            return res_list
        else:
            res_list = []
            subgraphs_per_child_list = []
            for dcmp_var,dcmp_node in node.dcmp_dict.items():
                child_res_list = get_deconstrain_decomposition_for_node(cur_graph,dcmp_node,retranslation_dict,redecomposition_dict,
                                                                    get_node_hash=get_node_hash,
                                                                    cur_cache=cur_cache,
                                                                    skip_set=skip_set,
                                                                    visited_set=visited_set.copy(),
                                                                    check_trivial=check_trivial,
                                                                    filter_res=filter_res,
                                                                    mode=mode)
                subgraphs_per_child_list.append(child_res_list)
            for child_subgraph_list in itertools.product(*subgraphs_per_child_list):
                assert len(child_subgraph_list) == len(node.dcmp_dict)
                new_node = copy_graph(node)
                i = 0
                for dcmp_var in new_node.dcmp_dict.keys():
                    new_node.dcmp_dict[dcmp_var] = child_subgraph_list[i]
                    child_subgraph_list[i].parent = new_node
                    dfs_translate(new_node,mode='NoRun',t_type='template')
                    i += 1
                res_list.append(new_node)
            visited_set.update([get_node_hash(entry) for entry in get_all_descendants(node)])
            return res_list
        
    if get_node_hash(node) not in cur_cache:
        cur_cache[get_node_hash(node)] = {}
    #to_constrain = classify_node_for_deconstrain(cur_graph,node)
    to_constrain = classify_node_for_deconstrain(cur_graph,node,visited_set=visited_set)
    if to_constrain not in cur_cache[get_node_hash(node)]:
        decomposition_list = get_possible_decomposition_for_node(node,redecomposition_dict)
        topk = 1
        select_idx_list = np.argsort([calc_decompose_coverage(node.assert_text,new_str_dcmp_dict) for new_str_dcmp_dict in decomposition_list])[::-1]
        decomposition_list = [decomposition_list[idx] for idx in select_idx_list[:topk]]
        cur_cache[get_node_hash(node)][to_constrain] = []
    else:
        return [copy_graph(entry) for entry in cur_cache[get_node_hash(node)][to_constrain]]

    visited_set.add(get_node_hash(node))
    for new_str_dcmp_dict in decomposition_list:
        iter_graph = copy_graph(cur_graph)
        iter_node = find_descendant(iter_graph,get_unique_node_id(node))
        iter_node.set_dcmp_dict(new_str_dcmp_dict,force_new=False)
        assert get_node_hash(node) == get_node_hash(iter_node)
        translation_list = get_retranslations_for_node(iter_node,retranslation_dict)
        if mode == "prune":
            select_list = get_deconstrain_translation_helper(translation_list,to_constrain)
        elif mode == "all":
            select_list = remove_duplicate_translations(translation_list)
        elif mode == "one_prune":
            select_list = [get_deconstrain_translation_helper(translation_list,to_constrain)[0]]
        else:
            assert False
        if len(iter_node.dcmp_dict) == 0:
            for new_translation in select_list:
                new_node = copy_graph(iter_node)
                new_node.translation = new_translation            
                cur_cache[get_node_hash(node)][to_constrain].append(new_node)
        else:
            for new_translation in select_list:
                subgraphs_per_child_list = []
                new_graph = copy_graph(iter_graph)
                new_node = find_descendant(new_graph,get_unique_node_id(node))
                new_node.template_translation = new_translation
                dfs_translate(new_graph,mode='NoRun',t_type='template')
                for dcmp_var,dcmp_node in new_node.dcmp_dict.items():
                    child_res_list = get_deconstrain_decomposition_for_node(new_graph,dcmp_node,retranslation_dict,redecomposition_dict,
                                                                        get_node_hash=get_node_hash,
                                                                        cur_cache=cur_cache,
                                                                        skip_set=skip_set,
                                                                        visited_set=visited_set.copy(),
                                                                        check_trivial=check_trivial,
                                                                        filter_res=filter_res,
                                                                        mode=mode)
                    subgraphs_per_child_list.append(child_res_list)
                #print("crossing num",np.prod([len(entry) for entry in subgraphs_per_child_list]))
                for child_subgraph_list in itertools.product(*subgraphs_per_child_list):
                    assert len(child_subgraph_list) == len(iter_node.dcmp_dict)
                    new_node = copy_graph(new_node)
                    i = 0
                    for dcmp_var in new_node.dcmp_dict.keys():
                        new_node.dcmp_dict[dcmp_var] = child_subgraph_list[i]
                        child_subgraph_list[i].parent = new_node
                        dfs_translate(new_node,mode='NoRun',t_type='template')
                        i += 1                   
                    cur_cache[get_node_hash(node)][to_constrain].append(new_node)
    

    if check_trivial:
        #print("before",len(cur_res_list))
        graph_translation_list = [g.translation for g in cur_cache[get_node_hash(node)][to_constrain]]
        trivial_translation_set = find_hold_set_rawconjmerge(graph_translation_list,formula_DUT="1")
        cur_cache[get_node_hash(node)][to_constrain] = [g for g in cur_cache[get_node_hash(node)][to_constrain] if g.translation not in trivial_translation_set]
        #print("after",len(cur_res_list))

    if len(cur_cache[get_node_hash(node)][to_constrain]) > 0 and filter_res and to_constrain is not None:
        cur_cache[get_node_hash(node)][to_constrain] = filter_graph_list(
                    cur_cache[get_node_hash(node)][to_constrain],
                    filter_func=get_most_constrained if to_constrain else get_least_constrained,
                )
    #print("after",len(cur_cache[get_node_hash(node)][to_constrain]))
    res_list = [copy_graph(entry) for entry in cur_cache[get_node_hash(node)][to_constrain]]
    return res_list

def get_deconstrain_graph(cur_graph,formula_DUT,retranslation_dict,redecomposition_dict,
                          cur_skip_set=set(),dec_cache={},check_graph=False,early_stop=False):
    
    dec_graph_list = get_deconstrain_decomposition_for_node(cur_graph,cur_graph,retranslation_dict,redecomposition_dict,
                                                          cur_cache=dec_cache,
                                                          skip_set=cur_skip_set,
                                                          visited_set=set())
    holds_graph_list = find_holds(dec_graph_list,formula_DUT,
    #holds_graph_list = get_holds_graph_list(dec_graph_list,formula_DUT,
                                            check_graph=False,
                                            early_stop=early_stop)
    if check_graph:
        res_graph_list = [g for g in holds_graph_list if len(check_graph_validity(g))==0]
    else:
        res_graph_list = holds_graph_list
    
    if len(res_graph_list) > 0:
        return res_graph_list

    if check_graph and len(holds_graph_list) > 0:
        trivial_graph_list = [holds_graph_list[0]]
    else:
        trivial_graph_list = find_holds(dec_graph_list,formula_DUT="1",check_graph=False,early_stop=True,remove_trivial=False)        
    if len(trivial_graph_list) > 0:
        trivial_graph = trivial_graph_list[0]
        cur_bad_nodes = get_depthfirst_trivial_translation(trivial_graph)
        bad_id_set = set([get_unique_node_id(entry) for entry in cur_bad_nodes])
        available_id_set = bad_id_set - cur_skip_set
        available_list = sort_by_num_descendants([entry for entry in cur_bad_nodes if get_unique_node_id(entry) in available_id_set])

        for cur_num in range(len(available_list)):
            for cur_update_list in itertools.combinations(available_list,cur_num+1):
                cur_update_id_set = set([get_unique_node_id(entry) for entry in cur_update_list])
                retranslations_per_node_list = []
                for node in cur_update_list:
                    select_list = get_constrain_retranslation_candidates_for_node(trivial_graph,node,retranslation_dict,inclusive=True)
                    retranslations_per_node_list.append(select_list)
                    
                for new_translation_list in itertools.product(*retranslations_per_node_list):
                    new_graph = get_translation_update(trivial_graph,cur_update_list,new_translation_list,get_node_id_func=get_unique_node_id)
                    next_bad_list = get_depthfirst_trivial_translation(new_graph)
                    next_bad_id_set = set([get_unique_node_id(entry) for entry in next_bad_list])
                    #if len(next_bad_id_set.intersection(bad_id_set)) == 0:
                    #if len(next_bad_id_set.intersection(bad_id_set)) < len(bad_id_set):
                    if len(next_bad_id_set.intersection(cur_update_id_set)) < len(cur_update_id_set):
                        new_skip_set = cur_update_id_set.union(cur_skip_set)
                        #new_skip_set = bad_id_set.union(cur_skip_set)
                        print("going down!",len(cur_update_list),len(next_bad_id_set),len(new_skip_set))
                        holds_graph_list = get_deconstrain_graph(new_graph,formula_DUT,retranslation_dict,redecomposition_dict,
                                                                 cur_skip_set=new_skip_set,
                                                                 #dec_cache=dec_cache,
                                                                 dec_cache={},#cannot reuse cache for now since caching only works if skip_set start from root
                                                                 check_graph=check_graph,
                                                                 early_stop=early_stop)
                        if len(holds_graph_list) > 0:
                            return holds_graph_list
    return []

def get_most_senior_node(node_list):
    res_list = []
    for node in node_list:
        is_child = False
        for i in range(len(res_list)):
            if node in get_all_descendants(res_list[i]):
                is_child = True
                break
            elif node in get_all_ancestors(res_list[i]):
                res_list[i] = node
        if not is_child:
            res_list.append(node)
        res_list = list(set(res_list))
    return res_list

def classify_node_for_constrain(cur_graph,node):
    true_ablate_formula, false_ablate_formula = get_ablate_formulae_for_node(cur_graph,node)
    is_contains_false  = spot_utils.check_formula_contains_formula(cur_graph.translation,false_ablate_formula)
    is_contains_true = spot_utils.check_formula_contains_formula(cur_graph.translation,true_ablate_formula)
    is_false_contains  = spot_utils.check_formula_contains_formula(false_ablate_formula,cur_graph.translation)
    is_true_contains = spot_utils.check_formula_contains_formula(true_ablate_formula,cur_graph.translation)
    #both true_ablate and false_ablate are subset of cur_translation
    if is_contains_true and is_contains_false and is_true_contains and is_false_contains:
        to_constrain = None
    elif is_contains_true and is_contains_false and not is_true_contains and is_false_contains:
        #true_ablate is a smaller subset of cur_translation, thus deconstrain the node to constrain the graph
        to_constrain = False
    elif is_contains_true and is_contains_false and is_true_contains and not is_false_contains:
        #false_ablate is a smaller subset of cur_translation, thus constrain the node to constrain the graph
        to_constrain = True
    elif is_contains_true and is_contains_false and not is_true_contains and not is_false_contains:
        #both false_ablate and true_ablate are smaller subset of cur_translation, thus it is not clearly a decontrain or constrain node
        to_constrain = None

    #false_ablate is a subset of cur_translation, true_ablate is not a subset of cur_translation
    elif not is_contains_true and is_contains_false and is_true_contains and is_false_contains:
        #true_ablate is a superset of cur_translation
        #false_ablate is equal to cur_translation, this should not be possible
        to_constrain = True
    elif not is_contains_true and is_contains_false and not is_true_contains and is_false_contains:
        #true_ablate is not a subset or superset of cur_translation
        #false_ablate is equal to cur_translation, this should not be possible
        assert False
    elif not is_contains_true and is_contains_false and is_true_contains and not is_false_contains:
        #true_ablate is a superset of cur_translation
        #false_ablate is a smaller subset of cur_translation
        #constrain the node to constrain the graph
        to_constrain = True
    elif not is_contains_true and is_contains_false and not is_true_contains and not is_false_contains:
        #true_ablate is not a subset or superset of cur_translation
        #false_ablate is a smaller subset of cur_translation
        to_constrain = True 

    #true_ablate is a subset of cur_translation, false_ablate is not a subset of cur_translation
    elif is_contains_true and not is_contains_false and is_true_contains and is_false_contains:
        #true_ablate is equal to the cur_translation, this should not be possible
        #false_ablate is a superset of cur_translation
        to_constrain = False
    elif is_contains_true and not is_contains_false and not is_true_contains and is_false_contains:
        #true_ablate is a smaller subset of cur_translation
        #false_ablate is a supserset of cur_translation
        to_constrain = False
    elif is_contains_true and not is_contains_false and is_true_contains and not is_false_contains:
        #true_ablate is equal to cur_translation, this should not be possible
        #false_ablate is not a subset or superset of cur_translation
        assert False
    elif is_contains_true and not is_contains_false and not is_true_contains and not is_false_contains:
        #true_ablate is a smaller subset of cur_translation
        #false_ablate is not a subset or superset of cur_translation
        to_constrain = True 
    
    #true_ablate is not a subset of cur_translation, false_ablate is not a subset of cur_translation
    elif not is_contains_true and not is_contains_false and is_true_contains and is_false_contains:
        #true_ablate is a superset of cur_translation
        #false_ablate is a superset of cur_translation
        assert False
    elif not is_contains_true and not is_contains_false and not is_true_contains and is_false_contains:
        #true_ablate is not a subset or superset of cur_translation
        #false_ablate is a superset of cur_translation
        to_constrain = None
    elif not is_contains_true and not is_contains_false and is_true_contains and not is_false_contains:
        #true_ablate is a superset of cur_translation
        #false_ablate is not a subset or superset of cur_translation
        to_constrain = None
    elif not is_contains_true and not is_contains_false and not is_true_contains and not is_false_contains:
        #true_ablate is not a subset or superset of cur_translation
        #false_ablate is not a subset or superset of cur_translation
        to_constrain = None
        
    else:
        assert False
    return to_constrain

def classify_node_for_deconstrain(cur_graph,node,visited_set=None):
    if visited_set is not None:
        this_graph = get_visited_abstraction(cur_graph,visited_set)
        this_node = find_descendant(this_graph,get_unique_node_id(node))
        #print(this_graph.translation)
        #print(this_node.translation)
    else:
        this_graph = cur_graph
        this_node = node
    true_ablate_formula, false_ablate_formula = get_ablate_formulae_for_node(this_graph,this_node)
    is_contains_false  = spot_utils.check_formula_contains_formula(this_graph.translation,false_ablate_formula)
    is_contains_true = spot_utils.check_formula_contains_formula(this_graph.translation,true_ablate_formula)
    is_false_contains  = spot_utils.check_formula_contains_formula(false_ablate_formula,this_graph.translation)
    is_true_contains = spot_utils.check_formula_contains_formula(true_ablate_formula,this_graph.translation)

    #assert not (is_contains_false and is_contains_true and is_false_contains and is_true_contains)
    if is_contains_false and is_contains_true and is_false_contains and is_true_contains:
        to_constrain = True
    elif is_contains_true and is_contains_false and not is_true_contains and is_false_contains:
        to_constrain = True
        #most constraint
    elif is_contains_true and is_contains_false and is_true_contains and not is_false_contains:
        to_constrain = False
        #least constraint
    elif is_contains_true and is_contains_false and not is_true_contains and not is_false_contains:
        to_constrain = None
    elif not is_contains_true and is_contains_false:
        to_constrain = False
        #least constrained
    elif is_contains_true and not is_contains_false:
        to_constrain = True
        #most constrained
    elif not is_contains_true and not is_contains_false and is_true_contains and is_false_contains:
        #do not know what to do here
        to_constrain = None
    elif not is_contains_true and not is_contains_false and not is_true_contains and is_false_contains:
        to_constrain = True
        #most constrained
    elif not is_contains_true and not is_contains_false and is_true_contains and not is_false_contains:
        #least constrained
        to_constrain = False
    elif not is_contains_true and not is_contains_false and not is_true_contains and not is_false_contains:
        #do not know what to do here
        #assert False
        to_constrain = None
    else:
        assert False
    #print(to_constrain)
    return to_constrain

def get_constrain_translation_helper(cur_translation,translation_list,to_constrain,inclusive=False):
    if to_constrain == True:
        res = get_more_constrained(cur_translation,translation_list,inclusive=inclusive)
    elif to_constrain == False:
        res = get_less_constrained(cur_translation,translation_list,inclusive=inclusive)
    elif to_constrain == None:
        res = remove_duplicate_translations(translation_list)
    else:
        assert False
    #print(to_constrain,res)
    return res

def get_more_constrained(cur_translation,translation_list,inclusive=False):
    if inclusive:
        cur_translation_list = translation_list + [cur_translation]
    else:
        cur_translation_list = translation_list
    cur_translation_list = remove_duplicate_translations(cur_translation_list)

    res_list = get_most_constrained(translation_list)
    #res_list = []
    for t in cur_translation_list:
        if (spot_utils.check_formula_contains_formula(cur_translation,t) or \
            (not spot_utils.check_formula_contains_formula(t,cur_translation) and not spot_utils.check_formula_contains_formula(cur_translation,t))) \
            and t not in res_list:
            res_list.append(t)
    return res_list

def get_less_constrained(cur_translation,translation_list,inclusive=False):
    if inclusive:
        cur_translation_list = translation_list + [cur_translation]
    else:
        cur_translation_list = translation_list
    cur_translation_list = remove_duplicate_translations(cur_translation_list)
    
    res_list = get_least_constrained(translation_list)
    #res_list = []
    for t in cur_translation_list:
        if (spot_utils.check_formula_contains_formula(t,cur_translation) or \
            (not spot_utils.check_formula_contains_formula(t,cur_translation) and not spot_utils.check_formula_contains_formula(cur_translation,t))) \
            and t not in res_list:
            #either t is less constrained or neither is a subset of one another
            res_list.append(t)
    return res_list

def remove_duplicate_translations(translation_list):
    cur_list = []
    for cur_translation in translation_list:
        is_duplicate = False
        for entry in cur_list:
            if spot_utils.check_equivalent(cur_translation,entry):
                is_duplicate = True
                break
        if not is_duplicate:
            cur_list.append(cur_translation)
    return cur_list

def get_least_constrained(translation_list):
    translation_set = set(translation_list)
    candidate_set = set([translation_set.pop()])
    count = 0
    for cur_translation in translation_set:
        count += 1
        to_remove = set()
        for cur_candidate in candidate_set:
            if spot_utils.check_formula_contains_formula(cur_translation,cur_candidate):
                to_remove.add(cur_candidate)
        found_contains = False
        if len(to_remove) == 0:
            for cur_candidate in candidate_set:
                if spot_utils.check_formula_contains_formula(cur_candidate,cur_translation):
                    found_contains = True
                    break
        if len(to_remove) > 0 or not found_contains:
            candidate_set.add(cur_translation)
        candidate_set = candidate_set - to_remove
    assert len(candidate_set) > 0, translation_list
    return list(candidate_set)

def get_most_constrained(translation_list):
    to_check = set(translation_list)
    possible_candidate_set = set(translation_list)
    while len(to_check) > 0:
        to_remove = set()
        cur_candidate = to_check.pop()
        for possible_candidate in (possible_candidate_set - set([cur_candidate])):
            if spot_utils.check_formula_contains_formula(possible_candidate,cur_candidate):
                to_remove.add(possible_candidate)
        possible_candidate_set = possible_candidate_set - to_remove
        to_check = to_check - to_remove
    return list(possible_candidate_set)


def get_deconstrain_translation_helper(translation_list,to_constrain):
    if to_constrain == True:
        res = get_most_constrained(translation_list)
    elif to_constrain == False:
        res = get_least_constrained(translation_list)
    elif to_constrain == None:
        #res = remove_duplicate_translations(get_most_constrained(translation_list) + get_least_constrained(translation_list))
        res = remove_duplicate_translations(translation_list)
    else:
        assert False
    #print(to_constrain,res,len(res))
    return res
        
def get_retranslations_for_node(node,retranslation_dict,remove_equivalent=False):
    translation_list = []
    
    #if len(node.dcmp_dict) > 0:
    #    for node_obj in retranslation_dict[get_abstract_node_id(node)]:
    #        translation_list.append(node_obj.template_translation)
    #else:
    #    for node_obj in retranslation_dict[get_abstract_node_id(node)]:
    #        translation_list.append(node_obj.translation)
    translation_list = [entry for entry in retranslation_dict[get_abstract_node_id(node)]]
    if remove_equivalent:
        translation_list = remove_duplicate_translations(translation_list)
    return translation_list

def get_leaf_nodes(cur_graph):
    res_list = [node for node in get_all_descendants(cur_graph) if len(node.dcmp_dict) == 0]
    return res_list

def get_translation_update(cur_graph,node_list,translation_list,get_node_id_func):
    new_graph = copy_graph(cur_graph)
    for i in range(len(node_list)):
        new_node = find_descendant_by_id(new_graph,get_node_id_func(node_list[i]),get_node_id_func=get_node_id_func)
        if len(new_node.dcmp_dict) == 0:
            new_node.translation = translation_list[i]
        else:
            new_node.template_translation = translation_list[i]
    dfs_translate(new_graph,mode='NoRun',t_type='template')
    assert not None in [node.translation for node in get_all_descendants(new_graph)]
    return new_graph

def sort_by_num_descendants(node_list):
    sort_idx = np.argsort([len(get_all_descendants(node)) for node in node_list])
    return [node_list[idx] for idx in sort_idx]

def abstractize_nodes_in_graph(abs_graph,abs_node_list):
    abs_var_symbol = "abs_var"
    assert abs_var_symbol not in abs_graph.translation
    node_to_abs = dict((abs_node_list[i],"_"+abs_var_symbol+str(i)+"_") for i in range(len(abs_node_list)))
    node_to_translation = dict((abs_node_list[i],abs_node_list[i].translation) for i in range(len(abs_node_list)))
    for node,cur_abs_var_symbol in node_to_abs.items():
        node.translation = cur_abs_var_symbol
        node.template_translation = None
        node.dcmp_dict = {}
    dfs_translate(abs_graph,mode='NoRun',t_type='template')
    return abs_graph, node_to_translation

def get_nodes_by_depth(cur_graph,depth,include_leaves=False):
    if include_leaves and (depth==0 or len(cur_graph.dcmp_dict) == 0):
        return [cur_graph]
    elif not include_leaves and depth==0:
        return [cur_graph]
    else:
        res_list = []
        for dcmp_var,dcmp_node in cur_graph.dcmp_dict.items():
            res_list += get_nodes_by_depth(dcmp_node,depth-1,include_leaves=include_leaves)
        return res_list

def get_nodes_breadthfirst(cur_graph):
    res_list = []
    work_list = [cur_graph]
    while len(work_list) > 0:
        cur_node = work_list.pop(0)
        res_list.append(cur_node)
        work_list += list(cur_node.dcmp_dict.values())
    return res_list

def get_graph_abs_by_breadthfirst(cur_graph,count):
    abs_graph = copy_graph(cur_graph)
    node_list = get_nodes_breadthfirst(abs_graph)
    abs_node_list = get_most_senior_node(node_list[count:])
    sort_idx = np.argsort([node.assert_text for node in abs_node_list])
    abs_node_list = [abs_node_list[idx] for idx in sort_idx] #necessary to ensure hashes collide
    abs_graph, node_to_translation = abstractize_nodes_in_graph(abs_graph,abs_node_list)
    return abs_graph, node_to_translation
    
def get_graph_abs_by_depth(cur_graph,count):
    abs_graph = copy_graph(cur_graph)
    abs_node_list = get_nodes_by_depth(abs_graph,depth=count,include_leaves=True)
    abs_graph, node_to_translation = abstractize_nodes_in_graph(abs_graph,abs_node_list)
    return abs_graph, node_to_translation

def get_graph_merge(graph_list,get_node_id_func,ret_graph=False,deconstrain=True):
    mindiff_node_ids = get_graph_diff(graph_list)
    if not ret_graph:
        return get_translation_merge_str(graph_list,mindiff_node_ids,get_node_id_func,deconstrain=deconstrain)
    else:
        #tmp = get_translation_merge_str(graph_list,mindiff_node_ids,get_node_id_func,deconstrain=deconstrain)
        res = get_translation_merge_graph(graph_list,mindiff_node_ids,get_node_id_func,deconstrain=deconstrain)
        #assert spot_utils.check_equivalent(res.translation,tmp)
        return res

def get_translation_merge_graph(graph_list,input_merge_node_id_list,get_node_id_func,deconstrain=True):
    if deconstrain:
        op_to_constrain = " & "
        op_to_deconstrain = " | "
    else:
        op_to_constrain = " | "
        op_to_deconstrain = " & "

    merge_node_id_list = input_merge_node_id_list
    
    to_constrain_list = []
    for i in range(len(merge_node_id_list)):
        node_id = merge_node_id_list[i]
        tmp_node = find_descendant_by_id(graph_list[0],node_id,get_node_id_func=get_node_id_func)
        cur_visited_ids = set([get_unique_node_id(entry) for entry in get_all_ancestors(tmp_node,inclusive=False)])
        abs_to_constrain = classify_node_for_deconstrain(graph_list[0],tmp_node,visited_set=cur_visited_ids)        
        to_constrain_list.append(abs_to_constrain)
    
    while None in to_constrain_list:
        to_remove = []
        for i in range(len(merge_node_id_list)):
            if to_constrain_list[i] is None:
                node_id = merge_node_id_list[i]
                tmp_node = find_descendant_by_id(graph_list[0],node_id,get_node_id_func=get_node_id_func)
                assert tmp_node.parent is not None
                cur_visited_ids = set([get_unique_node_id(entry) for entry in get_all_ancestors(tmp_node.parent,inclusive=False)])
                abs_to_constrain = classify_node_for_deconstrain(graph_list[0],tmp_node.parent,visited_set=cur_visited_ids)
                to_constrain_list.append(abs_to_constrain)
                merge_node_id_list.append(get_node_id_func(tmp_node.parent))
                to_remove += [get_node_id_func(dcmp_node) for dcmp_node in tmp_node.parent.dcmp_dict.values()]
                assert node_id in to_remove
        to_constrain_list = [to_constrain_list[i] for i in range(len(merge_node_id_list)) if merge_node_id_list[i] not in to_remove]
        merge_node_id_list = [merge_node_id_list[i] for i in range(len(merge_node_id_list)) if merge_node_id_list[i] not in to_remove]
    
            
    new_graph_list = [copy_graph(graph_list[0])]
    for list_idx in range(len(merge_node_id_list)):
        node_id = merge_node_id_list[list_idx]
        node_list = []
        translation_set = set()
        for cur_graph in graph_list:
            cur_node = find_descendant_by_id(cur_graph,node_id,get_node_id_func=get_node_id_func)
            if cur_node.translation not in translation_set:
                node_list.append(cur_node)
            translation_set.add(cur_node.translation)
        
        translation_list = []
        merge_node = find_descendant_by_id(new_graph_list[0],node_id,get_node_id_func=get_node_id_func)
        merge_node.dcmp_dict.clear()
        for i in range(len(node_list)):
            if len(node_list[i].dcmp_dict) > 0:
                cur_template_translation = node_list[i].template_translation
                for abs_var,dcmp_node in node_list[i].dcmp_dict.items():
                    new_abs_var = abs_var+"_"+str(i)+"_"
                    cur_template_translation = cur_template_translation.replace("_"+abs_var+"_","_"+new_abs_var+"_")
                    merge_node.dcmp_dict[new_abs_var] = dcmp_node
                translation_list.append(cur_template_translation)
            else:
                translation_list.append(node_list[i].translation)
        
        to_add_list = []
        for new_graph in new_graph_list:
            new_node = find_descendant_by_id(new_graph,node_id,get_node_id_func=get_node_id_func)
            new_node.dcmp_dict = dict((k,copy_graph(v)) for k,v in merge_node.dcmp_dict.items())
            for dcmp_node in new_node.dcmp_dict.values():
                dcmp_node.parent = new_node
            
            abs_to_constrain = to_constrain_list[list_idx]
            
            if abs_to_constrain is None:
                #None should be removed from to_constrain_list
                assert False, "found a none to_constrain in get_translation_merge_graph!"
                alter_new_graph = copy_graph(new_graph)
                alter_new_node = find_descendant_by_id(alter_new_graph,node_id,get_node_id_func=get_node_id_func)
                if len(merge_node.dcmp_dict) > 0:
                    new_node.template_translation = op_to_deconstrain.join(["("+translation+")" for translation in translation_list])
                    alter_new_node.dcmp_dict = dict((k,copy_graph(v)) for k,v in merge_node.dcmp_dict.items())
                    for dcmp_node in alter_new_node.dcmp_dict.values():
                        dcmp_node.parent = alter_new_node
                    alter_new_node.template_translation = op_to_constrain.join(["("+translation+")" for translation in translation_list])
                else:
                    new_node.translation = op_to_deconstrain.join(["("+translation+")" for translation in translation_list])
                    alter_new_node.translation = op_to_constrain.join(["("+translation+")" for translation in translation_list])
                to_add_list.append(alter_new_graph)
            elif abs_to_constrain:
                if len(merge_node.dcmp_dict) > 0:
                    new_node.template_translation = op_to_constrain.join(["("+translation+")" for translation in translation_list])
                else:
                    new_node.translation = op_to_constrain.join(["("+translation+")" for translation in translation_list])
            else:
                if len(merge_node.dcmp_dict) > 0:
                    new_node.template_translation = op_to_deconstrain.join(["("+translation+")" for translation in translation_list])
                else:
                    new_node.translation = op_to_deconstrain.join(["("+translation+")" for translation in translation_list])
        new_graph_list += to_add_list

    if len(new_graph_list) > 1:
        #print("found a none to_constrain in get_translation_merge_graph!")
        assert False, "found a none to_constrain in get_translation_merge_graph!"
        merge_graph = copy_graph(new_graph_list[0])
        merge_graph.dcmp_dict.clear()
        translation_list = []
        for i in range(len(new_graph_list)):
            if len(new_graph_list[i].dcmp_dict) > 0:
                cur_template_translation = new_graph_list[i].template_translation
                for abs_var,dcmp_node in new_graph_list[i].dcmp_dict.items():
                    new_abs_var = abs_var+"_"+str(i)+"_"
                    cur_template_translation = cur_template_translation.replace("_"+abs_var+"_","_"+new_abs_var+"_")
                    merge_graph.dcmp_dict[new_abs_var] = dcmp_node
                    dcmp_node.parent = merge_graph
                translation_list.append(cur_template_translation)
            else:
                translation_list.append(new_graph_list[i].translation)
        if len(merge_graph.dcmp_dict) > 0:
            merge_graph.template_translation = op_to_deconstrain.join(["("+translation+")" for translation in translation_list])
        else:
            merge_graph.translation = op_to_deconstrain.join(["("+translation+")" for translation in translation_list])
    else:
        merge_graph = new_graph_list[0]
    dfs_translate(merge_graph,mode='NoRun',t_type='template')

    return merge_graph

def get_translation_merge_str(graph_list,input_merge_node_id_list,get_node_id_func,deconstrain=True):
    if deconstrain:
        op_to_constrain = " & "
        op_to_deconstrain = " | "
    else:
        op_to_constrain = " | "
        op_to_deconstrain = " & "
    
    merge_node_id_list = input_merge_node_id_list
    
    to_constrain_list = []
    for i in range(len(merge_node_id_list)):
        node_id = merge_node_id_list[i]
        tmp_node = find_descendant_by_id(graph_list[0],node_id,get_node_id_func=get_node_id_func)
        cur_visited_ids = set([get_unique_node_id(entry) for entry in get_all_ancestors(tmp_node,inclusive=False)])
        abs_to_constrain = classify_node_for_deconstrain(graph_list[0],tmp_node,visited_set=cur_visited_ids)        
        to_constrain_list.append(abs_to_constrain)

    while None in to_constrain_list:
        to_remove = []
        for i in range(len(merge_node_id_list)):
            if to_constrain_list[i] is None:
                node_id = merge_node_id_list[i]
                tmp_node = find_descendant_by_id(graph_list[0],node_id,get_node_id_func=get_node_id_func)
                assert tmp_node.parent is not None
                cur_visited_ids = set([get_unique_node_id(entry) for entry in get_all_ancestors(tmp_node.parent,inclusive=False)])
                abs_to_constrain = classify_node_for_deconstrain(graph_list[0],tmp_node.parent,visited_set=cur_visited_ids)
                to_constrain_list.append(abs_to_constrain)
                merge_node_id_list.append(get_node_id_func(tmp_node.parent))
                to_remove += [get_node_id_func(dcmp_node) for dcmp_node in tmp_node.parent.dcmp_dict.values()]
                assert node_id in to_remove
        to_constrain_list = [to_constrain_list[i] for i in range(len(merge_node_id_list)) if not merge_node_id_list[i] in to_remove]
        merge_node_id_list = [merge_node_id_list[i] for i in range(len(merge_node_id_list)) if not merge_node_id_list[i] in to_remove]
        
    new_graph_list = [copy_graph(graph_list[0])]
    for list_idx in range(len(merge_node_id_list)):
        node_id = merge_node_id_list[list_idx]
        translation_list = []
        for cur_graph in graph_list:
            cur_node = find_descendant_by_id(cur_graph,node_id,get_node_id_func=get_node_id_func)
            translation_list.append(cur_node.translation)
        translation_list = list(set(translation_list))

        to_add_list = []
        for new_graph in new_graph_list:
            new_node = find_descendant_by_id(new_graph,node_id,get_node_id_func=get_node_id_func)
            abs_to_constrain = to_constrain_list[list_idx]
            if abs_to_constrain is None:
                #None should be removed from to_constrain_list
                assert False, "found a none to_constrain in get_translation_merge_graph!"
                new_translation = op_to_deconstrain.join(["("+translation+")" for translation in translation_list])
                alter_new_graph = copy_graph(new_graph)
                alter_new_node = find_descendant_by_id(alter_new_graph,node_id,get_node_id_func=get_node_id_func)
                alter_new_translation = op_to_constrain.join(["("+translation+")" for translation in translation_list])
                alter_new_node.dcmp_dict = {}
                alter_new_node.template_translation = None
                alter_new_node.translation = alter_new_translation
                to_add_list.append(alter_new_graph)
            elif abs_to_constrain:
                new_translation = op_to_constrain.join(["("+translation+")" for translation in translation_list])
            else:
                new_translation = op_to_deconstrain.join(["("+translation+")" for translation in translation_list])
            new_node.dcmp_dict = {}
            new_node.template_translation = None
            new_node.translation = new_translation
        new_graph_list += to_add_list
        for new_graph in new_graph_list:
            dfs_translate(new_graph,mode='NoRun',t_type='template')
    final_merge_translation = op_to_deconstrain.join(["("+new_graph.translation+")" for new_graph in new_graph_list])

    return final_merge_translation

def get_conjucts_by_abstraction(abs_conjuct_list,node_to_translation): 
    res_conjuct_list = []
    for abs_clause in abs_conjuct_list:
        res_clause = abs_clause
        for abs_node,translation in node_to_translation.items():
            res_clause = res_clause.replace(abs_node.translation,"("+translation+")")
        res_conjuct_list.append(res_clause)

    return res_conjuct_list


def construct_conjuct_hash_map(graph_list,get_graph_abs_func,drop_threshold=0,**kwargs):
    abs_to_g = {}
    g_to_abs = {}
    for cur_graph in graph_list:
        abs_graph, node_to_translation = get_graph_abs_func(cur_graph,**kwargs)
        g_to_abs[cur_graph] = (abs_graph,node_to_translation)
        if abs_graph.translation not in abs_to_g:
            abs_to_g[abs_graph.translation] = set([cur_graph])
        else:
            abs_to_g[abs_graph.translation].add(cur_graph)
    
    hash_table = {}
    g_to_count = {}    
    for abs_graph_translation in abs_to_g:
        abs_conjuct_list = spot_utils.get_conjucts(abs_graph_translation)
        #filter out trivially true clauses
        #abs_conjuct_list = get_nontrivialtrue(abs_conjuct_list)
        for cur_graph in abs_to_g[abs_graph_translation]:
            abs_graph,node_to_translation = g_to_abs[cur_graph]
            cur_conjuct_list = get_conjucts_by_abstraction(abs_conjuct_list,node_to_translation)
            cur_conjuct_set = set(cur_conjuct_list)
            g_to_count[cur_graph] = len(cur_conjuct_set)
            for clause in cur_conjuct_set:
                if clause not in hash_table:
                    #for e_clause in hash_table:
                    #    assert not spot_utils.check_equivalent(clause,e_clause)
                    hash_table[clause] = set([cur_graph])
                else:
                    hash_table[clause].add(cur_graph)
    
    to_remove_list = []
    for clause,g_set in hash_table.items():
        if len(g_set) <= drop_threshold:
            to_remove_list.append(clause)
    for clause in to_remove_list:
        for g in hash_table[clause]:
            g_to_count[g] -= 1
        del hash_table[clause]

    return hash_table, g_to_count

def get_best_conjuct_hash_map_by_depth(graph_list,**kwargs):
    best_score = float('-inf')
    max_count = get_max_depth(graph_list[0])
    best_hash_map = {}
    best_g_to_count = {}
    for count in range(max_count):
        new_hash_map,new_g_to_count = construct_conjuct_hash_map(graph_list,count=count,**kwargs)
        if len(new_hash_map) > 0:
            cur_score = np.sum([len(entry) for entry in new_hash_map.values()])/len(new_hash_map)
        else:
            cur_score = float('-inf')
        if cur_score > best_score:
            best_hash_map = new_hash_map
            best_g_to_count = new_g_to_count
            best_score = cur_score
    return best_hash_map, best_g_to_count

def construct_ideal_conjuct_hash_map(graph_list):
    hash_table = {}
    clause_list = []
    for cur_graph in tqdm(graph_list):
        cur_conjuct_list = spot_utils.get_conjucts(cur_graph.translation)
        cur_conjuct_set = set(cur_conjuct_list)
        for clause in cur_conjuct_set:
            cur_clause_hash = None
            for key_clause in clause_list:
                if spot_utils.check_equivalent(clause,key_clause):
                    cur_clause_hash = key_clause
                    break
            if cur_clause_hash is None:
                cur_clause_hash = clause
                hash_table[cur_clause_hash] = set([cur_graph])
                clause_list.append(clause)
            else:
                hash_table[cur_clause_hash].add(cur_graph)
    return hash_table, clause_list

def find_holds_by_hashmap(graph_list,hash_map,formula_DUT,check_graph=False,early_stop=False,remove_trivial=False,g_to_count=None):
    assert not (early_stop and remove_trivial) #for now, cannot do both
    active_set = set(graph_list)
    profile_time = 0
    while len(hash_map) > 0:
        largest_count = float('-inf')
        select_clause = None
        for clause in hash_map:
            assert len(hash_map[clause]) > 0
            if len(hash_map[clause]) > largest_count:
                largest_count = len(hash_map[clause])
                select_clause = clause
        to_remove = set()
        is_holds = spot_utils.check_formula_contains_formula(select_clause,formula_DUT)
        if not is_holds:
            to_remove = hash_map[select_clause]
        else:       
            if early_stop:
                for g in hash_map[select_clause]:
                    g_to_count[g] -= 1
                    if g_to_count[g] == 0:
                        holds_graph_list = get_holds_graph_list([g],formula_DUT,
                                                check_graph=check_graph,
                                                early_stop=early_stop)
                        if len(holds_graph_list) > 0:
                            return holds_graph_list, True
                        else:
                            to_remove.add(g)
            elif remove_trivial and spot_utils.check_formula_contains_formula(select_clause,"1"):
                for g in hash_map[select_clause]:
                    g_to_count[g] -= 1
                    assert g_to_count[g] >= 0
                    if g_to_count[g] == 0:
                        #assert spot_utils.check_equivalent(g.translation,"1")
                        to_remove.add(g)
        
        clause_to_remove = []
        if len(to_remove) > 0:
            active_set = active_set - to_remove
            for clause in hash_map:
                if clause != select_clause:
                    hash_map[clause] = hash_map[clause] - to_remove
                    if len(hash_map[clause]) == 0:
                        clause_to_remove.append(clause)
        del hash_map[select_clause]
        for clause in clause_to_remove:
            del hash_map[clause]
    active_list = list(active_set)
    return active_list, False

def find_holds_by_constrainmerge_hash_map(merge_hash_map,formula_DUT,conj_split=True):
    if len(merge_hash_map) == 0:
        return []
        
    merge_g_to_hash = {}
    for g_hash,g_set in merge_hash_map.items():
        merge_g = get_graph_merge(list(g_set),get_node_id_func=get_unique_node_id,ret_graph=True,deconstrain=False)
        merge_g_to_hash[merge_g] = g_hash
    merge_active_list = list(merge_g_to_hash.keys())

    if conj_split:
        conj_hash_map,g_to_count = get_best_conjuct_hash_map_by_depth(merge_active_list,
                                                                 #get_graph_hash=get_conjucts_by_abstraction,
                                                                 get_graph_abs_func=get_graph_abs_by_depth,
                                                                 drop_threshold=0) #important that drop_threshold=0
        
        merge_active_list,_ = find_holds_by_hashmap(merge_active_list,conj_hash_map,formula_DUT,early_stop=False)
    else:
        merge_active_list = [g for g in merge_active_list if spot_utils.check_formula_contains_formula(g.translation,formula_DUT)]
        
    #merge_active_list = set(merge_active_list) - set(to_remove_list)
    active_set = set()
    for merge_g in merge_active_list:
        g_hash = merge_g_to_hash[merge_g]
        active_set.update(merge_hash_map[g_hash])
    active_list = list(active_set)
    return active_list

def get_merge_hash_map(graph_list,node_sort_func,num_nodes):
    merge_hash_map = {}
    for cur_graph in graph_list:
        node_list = node_sort_func(cur_graph)[:num_nodes]
        g_hash = tuple([get_node_translation(node) for node in node_list])
        if g_hash not in merge_hash_map:
            merge_hash_map[g_hash] = set([cur_graph])
        else:
            merge_hash_map[g_hash].add(cur_graph)
    return merge_hash_map

def find_holds_by_merge_hash_map(merge_hash_map,formula_DUT,conj_split=True):
    if len(merge_hash_map) == 0:
        return []
    
    merge_g_to_hash = {}
    for g_hash,g_set in merge_hash_map.items():
        merge_g = get_graph_merge(list(g_set),get_node_id_func=get_unique_node_id,ret_graph=True)
        merge_g_to_hash[merge_g] = g_hash
    merge_active_list = list(merge_g_to_hash.keys())

    if conj_split:
        conj_hash_map,g_to_count = get_best_conjuct_hash_map_by_depth(merge_active_list,
                                                                 #get_graph_hash=get_conjucts_by_abstraction,
                                                                 get_graph_abs_func=get_graph_abs_by_depth,
                                                                 #drop_threshold=1,
                                                                 drop_threshold=0)
        
        merge_active_list,_ = find_holds_by_hashmap(merge_active_list,conj_hash_map,formula_DUT,
                                                                          check_graph=True,
                                                                          early_stop=False)
    else:
        merge_active_list = [g for g in merge_active_list if spot_utils.check_formula_contains_formula(g.translation,formula_DUT)]
    
    active_set = set()
    for merge_g in merge_active_list:
        g_hash = merge_g_to_hash[merge_g]
        active_set.update(merge_hash_map[g_hash])
    active_list = list(active_set)
    return active_list

def find_holds(graph_list,formula_DUT,check_graph=False,early_stop=False,remove_trivial=True,conj_split=True,disj_merge=True):
    if len(graph_list) == 0:
        return []
    active_list = graph_list
    max_num_nodes = np.max([len(get_all_descendants(g)) for g in graph_list])
    min_merge_ratio = 0.1
    if disj_merge:
        cur_num_nodes = 1
    else:
        cur_num_nodes = max_num_nodes
    merge_hash_map = get_merge_hash_map(active_list,node_sort_func=get_nodes_breadthfirst,num_nodes=cur_num_nodes)
    cur_map_size = len(merge_hash_map)
    while len(active_list) > 0:
        start_t = time.time()
        cur_num_active = len(active_list)
        for next_num_nodes in range(cur_num_nodes+1,max_num_nodes+1):
            new_merge_hash_map = get_merge_hash_map(active_list,node_sort_func=get_nodes_breadthfirst,num_nodes=next_num_nodes)
            merge_hash_map = new_merge_hash_map
            cur_num_nodes = next_num_nodes
            if len(new_merge_hash_map) > cur_map_size and len(new_merge_hash_map)/len(active_list) >= min_merge_ratio:
                break
        cur_num_g = len(merge_hash_map)
        print(cur_num_active,"num merge graphs:",cur_num_g,"cur num nodes:",cur_num_nodes,"max:",max_num_nodes)
        
        if remove_trivial:
            to_remove_list = find_holds_by_constrainmerge_hash_map(merge_hash_map,formula_DUT="1",conj_split=conj_split)
            active_list = list(set(active_list) - set(to_remove_list))
            merge_hash_map = get_merge_hash_map(active_list,node_sort_func=get_nodes_breadthfirst,num_nodes=cur_num_nodes)
        active_list = find_holds_by_merge_hash_map(merge_hash_map,formula_DUT,conj_split=conj_split)

        if early_stop:
            merge_hash_map = get_merge_hash_map(active_list,node_sort_func=get_nodes_breadthfirst,num_nodes=cur_num_nodes)
            holds_list = find_holds_by_constrainmerge_hash_map(merge_hash_map,formula_DUT,conj_split=conj_split)
            if len(holds_list) > 0:
                to_remove_list = find_holds(holds_list,formula_DUT="1",remove_trivial=False,early_stop=False,conj_split=conj_split)
                holds_list = list(set(holds_list)-set(to_remove_list))
                if check_graph:
                    holds_list = [g for g in holds_list if len(check_graph_validity(g))==0]
                if len(holds_list) > 0:
                    return holds_list
        end_t = time.time()
        #print(len(active_list),end_t-start_t)
        
        if cur_num_g == cur_num_active or cur_num_nodes == max_num_nodes:
            break
        else:
            cur_map_size = len(get_merge_hash_map(active_list,node_sort_func=get_nodes_breadthfirst,num_nodes=cur_num_nodes))
    if check_graph:
        active_list = [g for g in active_list if len(check_graph_validity(g))==0]
    return active_list

def get_constrain_retranslation_candidates_for_node(cur_graph,node,retranslation_dict,inclusive=True):
    to_constrain = classify_node_for_constrain(cur_graph,node)
    if len(node.dcmp_dict) > 0:
        cur_translation = node.template_translation
    else:
        cur_translation = node.translation
    #print("to_constrain:",to_constrain)
    translation_list = get_retranslations_for_node(node,retranslation_dict)
    select_list = get_constrain_translation_helper(cur_translation,translation_list,to_constrain,inclusive=inclusive)
    return select_list

def filter_graph_list(graph_list,filter_func,**kwargs):
    if len(graph_list) == 0:
        return []
    translation_list = [entry.translation for entry in graph_list]
    filtered_list = filter_func(translation_list,**kwargs)
    res_list = [entry for entry in graph_list if entry.translation in filtered_list]
    return res_list

def get_holds_graph_list(graph_list,formula_DUT,check_graph=False,early_stop=False):
    res_list = []
    for test_graph in tqdm(graph_list,disable=False):
        if not spot_utils.check_formula_contains_formula(test_graph.translation,"1",use_contains_split=True) \
            and spot_utils.check_formula_contains_formula(test_graph.translation,formula_DUT,use_contains_split=True) \
            and (not check_graph or len(check_graph_validity(test_graph))==0):
            #print('yay!',spot_utils.check_equivalent(test_graph.translation,formula_DUT),len(check_graph_validity(test_graph)))
            res_list.append(test_graph)
            if early_stop:
                break
    return res_list

def get_conj_dict(translation_list,depth=2):
    conj_dict = {}
    g_to_count = {}
    data = []
    for g in translation_list:
        cur_list = spot_utils.get_conjucts(g,depth=depth)
        g_to_count[g] = len(cur_list)
        data.append(len(cur_list))
        for entry in cur_list:
            if entry not in conj_dict:
                conj_dict[entry] = [g]
            else:
                conj_dict[entry].append(g)
    return conj_dict, g_to_count

def run_batch_mc(translation_list,conj_dict,g_to_count,formula_DUT,early_stop=False):
    disj_clause_dict = {}
    
    if formula_DUT in disj_clause_dict:
        disj_list = disj_clause_dict[formula_DUT]
    else:
        disj_list = [formula_DUT]
    
    active_set = set(translation_list)
    while len(conj_dict) > 0:
        to_remove = []
        to_delete = []
        for clause in conj_dict:
            to_delete.append(clause)
            found_not_hold = False
            for d_clause in disj_list:
                if not spot_utils.check_formula_contains_formula(clause,d_clause):
                    to_remove = conj_dict[clause]
                    found_not_hold = True
                    break
            if found_not_hold:
                break
            elif early_stop:
                for g in conj_dict[clause]:
                    g_to_count[g] -= 1
                    if g_to_count[g] == 0:
                        return [g]
        if len(to_remove) > 0:
            to_remove_set = set(to_remove)
            active_set = active_set - to_remove_set
            for clause in conj_dict:
                conj_dict[clause] = list(set(conj_dict[clause])-to_remove_set)
                if len(conj_dict[clause]) == 0:
                    to_delete.append(clause)
        for clause in set(to_delete):
            del conj_dict[clause]
    return active_set    

def find_hold_set_rawconjmerge(translation_list,formula_DUT,early_stop=False):
    conj_dict, g_to_count = get_conj_dict(translation_list,depth=1)
    return run_batch_mc(translation_list,conj_dict,g_to_count,formula_DUT,early_stop=early_stop)

def get_conjmerge_holds_graph_list(graph_list,formula_DUT,check_graph=False,early_stop=False):
    assert not check_graph, "check graph is inefficient, disable for now"
    translation_list = [g.translation for g in graph_list]
    trivial_set = find_hold_set_rawconjmerge(translation_list,"1")
    active_set = find_hold_set_rawconjmerge(list(set(translation_list)-trivial_set),formula_DUT,early_stop=early_stop)
    return [g for g in graph_list if g.translation in active_set]

def filter_by_variable_names(cur_graph,node_list,formula_DUT):
    targ_var_set = set(spot_utils.get_variables(formula_DUT))
    res_list = []
    for node in node_list:
        node_var_set = set(spot_utils.get_variables(node.translation))
        if len(node_var_set.intersection(targ_var_set)):
            res_list.append(node)
    return res_list

def get_ablate_culprit_batch(graph_list,formula_DUT):
    all_translation_list = []
    for g in graph_list:
        for node in get_all_descendants(g):
            true_ablate_f, false_ablate_f = get_ablate_formulae_for_node(g,node)
            all_translation_list.append(true_ablate_f)
            all_translation_list.append(false_ablate_f)
    hold_translation_set = find_hold_set_rawconjmerge(all_translation_list,formula_DUT=formula_DUT)
    res_list = []
    i = 0
    for g in graph_list:
        possible_dontcare_node_list = []
        possible_dontcare_translation_list = []
        for node in get_all_descendants(g):
            if all_translation_list[i] not in hold_translation_set \
                and all_translation_list[i+1] not in hold_translation_set:
                possible_dontcare_node_list.append(node)
                possible_dontcare_translation_list.append(all_translation_list[i])
                possible_dontcare_translation_list.append(all_translation_list[i+1])
            i += 2
        graph_contain_set = find_hold_set_rawconjmerge(possible_dontcare_translation_list,formula_DUT=g.translation)
        dontcare_list = []
        dontcare_mode = []
        j = 0
        for node in possible_dontcare_node_list:
            if node.translation != "0" and node.translation != "1" and \
                ((possible_dontcare_translation_list[j] in graph_contain_set) != (possible_dontcare_translation_list[j+1] in graph_contain_set)):
                if (possible_dontcare_translation_list[j] in graph_contain_set):
                    dontcare_list.append(node)
                    dontcare_mode.append("1")
                elif (possible_dontcare_translation_list[j+1] in graph_contain_set):
                    dontcare_list.append(node)
                    dontcare_mode.append("0")
            j += 2
        #cur_culprit_list = [node for node in get_all_descendants(g) if not node in dontcare_list]
        select_idx = [k for k in range(len(dontcare_list)) if set(get_all_descendants(dontcare_list[k])).issubset(set(dontcare_list))]
        dontcare_list = [dontcare_list[idx] for idx in select_idx]
        dontcare_mode = [dontcare_mode[idx] for idx in select_idx]
        if len(dontcare_list) > 0:
            most_senior_dc_list = get_most_senior_node(dontcare_list)
            most_senior_dc_mode = [dontcare_mode[dontcare_list.index(node)] for node in most_senior_dc_list]
            largest_idx = np.argmax([len(get_all_descendants(node)) for node in most_senior_dc_list])
            dontcare_node = most_senior_dc_list[largest_idx]
            new_graph = copy_graph(g)
            new_node = find_descendant(new_graph,get_unique_node_id(dontcare_node))
            new_node.translation = most_senior_dc_mode[largest_idx]
            new_node.dcmp_dict = {}
            dfs_translate(new_graph,mode='NoRun',t_type='template')
            assert spot_utils.check_formula_contains_formula(new_graph.translation,g.translation,use_contains_split=True)
            new_culprit_list = get_ablate_culprit_batch([new_graph],formula_DUT)[0]
            new_culprit_ids = [get_unique_node_id(node) for node in new_culprit_list if node != new_node]
            cur_culprit_list = [node for node in get_all_descendants(g) if get_unique_node_id(node) in new_culprit_ids]
            #cur_culprit_list = [node for node in get_all_descendants(g) if not node in get_all_descendants(dontcare_node)]
        else:
            cur_culprit_list = [node for node in get_all_descendants(g)]
        res_list.append(cur_culprit_list)
    return res_list

def check_decomposition_parsable(node,prediction):
    try:
        pred_dict = json.loads(prediction)
    except Exception as e:
        error_msg = "\nThe output must be JSON parsable"
        return f"Output cannot be parsed as JSON! error message: {e}" + error_msg
    
    for col_name in node.decompose_fewshots[0].keys():
        error_msg = "\nThe output must include each of the following" + str(list(node.decompose_fewshots[0].keys()))
        if col_name not in pred_dict:
            return "Output does not contain the"+col_name+" field." + error_msg
    
    if not isinstance(pred_dict["Decomposition"],dict):
        try:
            dcmp_dict = json.loads(pred_dict["Decomposition"])
        except Exception as e:
            error_msg = "\nThe Decomposition must be JSON parsable dictionary."
            return f"Decomposition cannot be parsed as JSON! error message: {e}" + error_msg
        if not isinstance(dcmp_dict,dict):
            error_msg = "\nThe Decomposition must be JSON parsable dictionary."
            return error_msg

    return "" #check passed

def check_decomposition_semantics(node,prediction):
    pred_dict = json.loads(prediction)
    
    if not isinstance(pred_dict["Decomposition"],dict):
        dcmp_dict = json.loads(pred_dict["Decomposition"])
    else:
        dcmp_dict = pred_dict["Decomposition"]
    
    dcmp_nl = pred_dict["Decomposed Natural language"]
    
    for abs_var in dcmp_dict:
        #if not is_str_in_str(dcmp_dict[abs_var],node.assert_text):
        if dcmp_dict[abs_var] not in node.assert_text:
            error_msg = "\nAll decompositions must be substrings of the Natural language phrase"
            return "\nDecomposition \"" + dcmp_dict[abs_var] + "\" is not in the Natural language phrase" + error_msg
    
    for abs_var in dcmp_dict:        
        for abs_var2 in dcmp_dict:
            if abs_var == abs_var2:
                continue
            else:
                error_msg = "\nDecompositions cannot be substrings of one another."
                if dcmp_dict[abs_var] in dcmp_dict[abs_var2]:
                    error_msg += " Either abstract variable " + abs_var + " should be removed or a new decomposition should be proposed." 
                    return "\nDecomposition corresponding to " +abs_var+" is a substring of another Decomposition corresponding to " + abs_var2 + error_msg
                if dcmp_dict[abs_var2] in dcmp_dict[abs_var]:
                    error_msg += " Either abstract variable " + abs_var2 + " should be removed or a new decomposition should be proposed." 
                    return "\nDecomposition corresponding to " +abs_var2+" is a substring of another Decomposition corresponding to " + abs_var + error_msg

    return "" #check passed

def check_decomposition(node,prediction):
    msg = check_decomposition_parsable(node,prediction)
    if msg != "":
        #msg is empty if pass
        return msg
    msg = check_decomposition_semantics(node,prediction)
    if msg != "":
        print(msg)
    return msg
    
def create_decompose_prompt(node):
    f = open("decomposition_prefix_prompt.txt","r")
    prefix = f.read()
    f.close()
    prefix_prompt = prefix
    prefix_prompt += "\nExample Decompositions:\n"
    for ex in node.decompose_fewshots:
        prefix_prompt += "\n" +  json.dumps(ex) + "\n"
    query_txt = "\nNote that you should avoiding decomposing the natural language into small phrases dependent phrases."
    query_txt += "\nConsider decomposing (" +",".join(node.decompose_fewshots[0].keys())+ ") following natural language phrase (the format must be in the same JSON format as shown in previous examples): "
    #cur_prompt = prefix_prompt + query_txt + repr(node.assert_text)
    cur_prompt = prefix_prompt + query_txt + "{\"Natural language\":\""+ repr(node.assert_text)[1:-1]+"\"}"
    return cur_prompt

def postprocess_decompose(node,pred):
    res = pred
    try:
        res = json.loads(pred)
    except:
        return res
    
    if 'Decomposition' in res and not isinstance(res["Decomposition"],dict):
        try:
            res['Decomposition'] = json.loads(res["Decomposition"])
        except:
            pass
    dcmp_dict = res['Decomposition']
    dcmp_list = list(dcmp_dict.values())
    dcmp_dict = dict(("SYMBOL"+str(i),dcmp_list[i]) for i in range(len(dcmp_list)))
    res = dict((k,v) for k,v in dcmp_dict.items() if v != "" and v != node.assert_text)
    return res

def decompose_LLM(node,max_try=5):
    cur_prompt = create_decompose_prompt(node)
    pred,is_pass = get_checked_prediction(cur_prompt,node,check_decomposition,max_try=max_try,
                                          model="gpt-3.5-turbo-0125"
                                          #model="gpt-4-0125-preview" 
                                         )
    #response = get_inference_response(cur_prompt,model="gpt-3.5-turbo-0125")
    #pred = response["choices"][0]["message"]["content"]
    #print(is_pass)
    #print(cur_prompt)
    #print(pred)
    if is_pass:
        res = postprocess_decompose(node,pred)
    else:
        res = {}
    #print("valid dcmp:",is_pass,res)
    return res

def get_common_node_id_list(graph_list):
    nodeshare_count = {}
    for cur_graph in graph_list:
        assert len(set([get_unique_node_id(node) for node in get_all_descendants(cur_graph)])) == len(get_all_descendants(cur_graph))
        for node in get_all_descendants(cur_graph):
            node_id = get_unique_node_id(node)
            if node_id not in nodeshare_count:
                nodeshare_count[node_id] = 1
            else:
                nodeshare_count[node_id] += 1
    
    common_node_id_list = []
    for node_id,count in nodeshare_count.items():
        if count == len(graph_list):
            common_node_id_list.append(node_id)
    return common_node_id_list

def get_node_template_hash(node):
    cur_hash = get_node_translation(node)
    for dcmp_var,dcmp_node in node.dcmp_dict.items():
        cur_hash = cur_hash.replace("_"+dcmp_var+"_",dcmp_node.assert_text)
    return cur_hash
    
def get_graph_diff(graph_list):
    #find nodes which have same decomposition but different translation
    common_node_id_set = set(get_common_node_id_list(graph_list))
    common_root_set = set()
    for node_id in common_node_id_set:
        all_node_list = [find_descendant(g,node_id) for g in graph_list]
        all_node_hashes = set([get_node_template_hash(node) for node in all_node_list])
        if len(all_node_hashes) == 1:
            common_root_set.add(node_id)
    diff_node_id_set = common_node_id_set - common_root_set
    mindiff_node_ids = []
    for node_id in diff_node_id_set:
        node = find_descendant(graph_list[0],node_id)
        if node.parent is None or get_unique_node_id(node.parent) in common_root_set:
            mindiff_node_ids.append(node_id)
        #all_node_list = [find_descendant(g,node_id) for g in graph_list]
        #all([node.parent is None or get_unique_node_id(node.parent) in common_root_set for node in all_node_list])
    return mindiff_node_ids

def search_holds_graph(root,formula_DUT,retranslation_dict,redecomposition_dict):
    dec_graph_list = get_deconstrain_decomposition_for_node(root,root,retranslation_dict,redecomposition_dict,
                                              cur_cache={},
                                              skip_set=set(),
                                              visited_set=set(),
                                              mode="prune",
                                              check_trivial=False,
                                              filter_res=False)
    return [], dec_graph_list

def graph_to_decompose_cache(cur_graph):
    cur_decompose_cache = {}
    for node in get_all_descendants(cur_graph):
        if get_literal_node_id(node) not in cur_decompose_cache:
            cur_decompose_cache[get_literal_node_id(node)] = []
        cur_decompose_cache[get_literal_node_id(node)].append(dict((k,v.assert_text) for k,v in node.dcmp_dict.items()))
    return cur_decompose_cache

def get_node_dcmpdict_to_str(node):
    return dict((k,v.assert_text) for k,v in node.dcmp_dict.items())

def constrain_decomposition_for_node(cur_graph,node,retranslation_dict,redecomposition_dict,formula_DUT,get_node_hash=get_unique_node_id,
                                     dec_cache={},visited_set=set(),checked_graphs=set()):
    assert get_unique_node_id(node) not in visited_set

    all_dec_graph_list = []
    visited_set.add(get_unique_node_id(node))
    decomposition_list = get_possible_decomposition_for_node(node,redecomposition_dict)
    for new_str_dcmp_dict in decomposition_list:
        iter_graph = copy_graph(cur_graph)
        iter_node = find_descendant(iter_graph,get_unique_node_id(node))
        iter_node.set_dcmp_dict(new_str_dcmp_dict)
    
        select_list = get_retranslations_for_node(iter_node,retranslation_dict,remove_equivalent=True)
        
        for new_translation in tqdm(select_list,disable=True):
            new_graph = copy_graph(iter_graph)
            new_node = find_descendant(new_graph,get_unique_node_id(node))
            if len(new_node.dcmp_dict) > 0:
                new_node.template_translation = new_translation
            else:
                new_node.translation = new_translation
            dfs_translate(new_graph,mode='NoRun',t_type='template')
            if get_node_translation(node) is None \
                or not spot_utils.check_equivalent(get_node_translation(new_node),get_node_translation(node)) \
                or get_node_dcmpdict_to_str(new_node) != get_node_dcmpdict_to_str(node):
                
                dec_graph_list = get_deconstrain_decomposition_for_node(new_graph,new_graph,retranslation_dict,redecomposition_dict,
                                                         get_node_hash=get_node_hash,
                                                         skip_set=visited_set,
                                                         cur_cache=dec_cache,
                                                         visited_set=set())
                checked_graphs.update([g.translation for g in dec_graph_list])
                holds_graph_list = get_conjmerge_holds_graph_list(dec_graph_list,formula_DUT)

                if len(holds_graph_list) > 0:
                    all_dec_graph_list.append(holds_graph_list[0])
                else:
                    graph_translation_list = [g.translation for g in dec_graph_list]
                    trivial_set = find_hold_set_rawconjmerge(graph_translation_list,"1")
                    trivial_graph_list = [g for g in dec_graph_list if g.translation in trivial_set]
                    if len(trivial_graph_list) == len(graph_translation_list):
                        all_dec_graph_list.append(trivial_graph_list[0])
            else:
                all_dec_graph_list.append(cur_graph)
    if len(all_dec_graph_list) == 0:
        return []
    assert not any([get_node_translation(this_node) is None for this_node in get_all_descendants(all_dec_graph_list[0])])

    mc_graph_list = all_dec_graph_list
    

    res_list = []
    cur_visited_set = set()
    for entry in mc_graph_list:
        new_node = find_descendant(entry,get_unique_node_id(node))
        if len(new_node.dcmp_dict) > 0:
            cur_res_list = [entry]
            cur_visited_set = visited_set.copy()
            for dcmp_var,dcmp_node in new_node.dcmp_dict.items():
                new_res_list = []
                accum_visited_set = set()
                for i in range(len(cur_res_list)):
                    tmp_visited_set = cur_visited_set.copy()
                    cur_dcmp_node = find_descendant(cur_res_list[i],get_unique_node_id(dcmp_node))
                    new_res_list += constrain_decomposition_for_node(cur_res_list[i],cur_dcmp_node,retranslation_dict,redecomposition_dict,formula_DUT,
                                           get_node_hash=get_node_hash,
                                           dec_cache=dec_cache,
                                           visited_set=tmp_visited_set,
                                           checked_graphs=checked_graphs)
                    accum_visited_set.update(tmp_visited_set)
                cur_visited_set = accum_visited_set
                cur_res_list = new_res_list
            res_list += cur_res_list
        else:
            res_list += [entry]
    visited_set.update(cur_visited_set)
    return res_list

##define context retrieval method
data_filepath = 'full_amba.txt'
passages = context_retriever.get_sentences(data_filepath)
retriever = context_retriever.ContextRetriever(passages,mode="RAG")

##define ground truth formula here
translate_dict = create_translate_dict('rawcontext_decomposition-slave.xlsx')
formula_DUT = translate_dict[open("specs/amba_slave_godhal.txt").read()]