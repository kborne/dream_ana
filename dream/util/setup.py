import os, sys, glob, argparse
import yaml, copy
from types import SimpleNamespace
from psana import DataSource

from typing import Tuple, List, Dict, Any

import numpy as np
from collections import deque

def check_detectors(
    yaml_a: Dict[str, Any],
    yaml_b: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any], Dict[str, Dict[str, List[str]]]]:
    # 1) build detector → { prefix: [trailing…] }
    det_maps: Dict[str, Dict[str, List[str]]] = {
        det: (conf.get('return') if isinstance(conf.get('return'), dict) else {})
        for det, conf in yaml_b.items()
    }

    def split_var(v: str) -> Tuple[str, str]:
        return tuple(v.split(':', 1)) if ':' in v else (v, v)

    def as_list(field):
        if isinstance(field, list):
            return field
        if isinstance(field, str):
            return [field]
        return []

    # 2) deep‐copy A for pruning
    updated = copy.deepcopy(yaml_a)
    missing: List[str] = []

    # 3) drop plots exactly as before
    for pname, pconf in list(updated.get('plots', {}).items()):
        to_check: List[str] = []
        to_check += as_list(pconf.get('var'))
        to_check += list(pconf.get('arange', {}).keys())
        to_check += as_list(pconf.get('scan'))
        to_check += as_list(pconf.get('norm'))
       
        if 'func' in pconf.get('type', '') or 'func' in pconf.get('plot_type', ''):

            for key_func in pconf.keys():
           
                if key_func[:4] == 'func':
                    to_check += as_list(pconf[key_func].get('args1'))

        bad = []
        for vs in to_check:
            pre, post = split_var(vs)
            if not any(post in det_maps[d].get(pre, []) for d in det_maps):
                bad.append(vs)

        if bad:
            missing.extend(bad)
            updated['plots'].pop(pname, None)

    # 4) prune data section, handling both normal modes and the special 'x' mode
    for mode_name, mode in updated.get('data', {}).items():
        if mode_name == 'x':
            # handle data/x/scan, epics, bld, timing
            for dtype in ['scan', 'epics', 'bld', 'timing', 'fzp', 'atm']:
                if dtype in mode:
                    vals = as_list(mode[dtype])
                    new_vals = []
                    for post in vals:
                        if any(post in det_map.get(dtype, []) for det_map in det_maps.values()):
                            new_vals.append(post)
                        else:
                            missing.append(f"{dtype}:{post}")
                    if new_vals:
                        mode[dtype] = new_vals
                    else:
                        mode.pop(dtype, None)
        else:
            # existing code for data/<mode>/<prefix>
            for prefix, item in mode.items():
                # fvar unchanged except pruning
                if isinstance(item, dict) and 'fvar' in item:
                    new_f = []
                    for post in item['fvar']:
                        if any(post in det_maps[d].get(prefix, []) for d in det_maps):
                            new_f.append(post)
                        else:
                            missing.append(f"{prefix}:{post}")
                    item['fvar'] = new_f

                # normalize & prune 'var'
                vals = as_list(item.get('var'))
                new_v = []
                for post in vals:
                    if any(post in det_maps[d].get(prefix, []) for d in det_maps):
                        new_v.append(post)
                    else:
                        missing.append(f"{prefix}:{post}")
                if new_v:
                    item['var'] = new_v
                else:
                    item.pop('var', None)

    if missing:
        print("The following vars are missing:", ", ".join(sorted(set(missing))))

    # 5) build requested_vars_by_detector, now including data/x lists
    requested: Dict[str, Dict[str, List[str]]] = {}
    for det, ret_map in det_maps.items():
        pref_to_trs: Dict[str, List[str]] = {}

        # from plots (unchanged)
        for pconf in updated.get('plots', {}).values():
            for vs in as_list(pconf.get('var')):
                pre, post = split_var(vs)
                if post in ret_map.get(pre, []):
                    pref_to_trs.setdefault(pre, []).append(post)
            for ak in pconf.get('arange', {}):
                pre, post = split_var(ak)
                if post in ret_map.get(pre, []):
                    pref_to_trs.setdefault(pre, []).append(post)
            for vs in as_list(pconf.get('scan')) + as_list(pconf.get('norm')):
                pre, post = split_var(vs)
                if post in ret_map.get(pre, []):
                    pref_to_trs.setdefault(pre, []).append(post)


            # only if this is a func‐plot, harvest its args1
            if 'func' in pconf.get('type', '') or 'func' in pconf.get('plot_type', ''):
                for key_func, func_conf in pconf.items():
                    if key_func.startswith('func') and isinstance(func_conf, dict):
                        for vs in as_list(func_conf.get('args1', [])):
                            pre, post = split_var(vs)
                            if post in ret_map.get(pre, []):
                                pref_to_trs.setdefault(pre, []).append(post)        

        # from data
        for mode_name, mode in updated.get('data', {}).items():
            if mode_name == 'x':
                # each dtype is a simple list
                for dtype, vals in mode.items():
                    for post in as_list(vals):
                        if post in ret_map.get(dtype, []):
                            pref_to_trs.setdefault(dtype, []).append(post)
            else:
                # original data/<mode>/<prefix> handling
                for prefix, item in mode.items():
                    for post in item.get('fvar', []):
                        if post in ret_map.get(prefix, []):
                            pref_to_trs.setdefault(prefix, []).append(post)
                    for post in as_list(item.get('var')):
                        if post in ret_map.get(prefix, []):
                            pref_to_trs.setdefault(prefix, []).append(post)

        # dedupe
        for pre, lst in pref_to_trs.items():
            seen = set()
            uniq = []
            for x in lst:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            pref_to_trs[pre] = uniq

        if pref_to_trs:
            requested[det] = pref_to_trs

    needed = list(requested.keys())
    return needed, updated, requested

def parse_boolean(value):
    value = value.lower()

    if value in ["True","true", "yes", "y", "1", "t"]:
        return True
    elif value in ["False","false", "no", "n", "0", "f"]:
        return False

    return False


def read_args():

    parser = argparse.ArgumentParser(description='preproc')
    parser.add_argument(
        '--exp',
        metavar='NAME',
        type=str,
        default=None,
        help='(optional) name of the experiment'
    )
    parser.add_argument(
        '--run',
        metavar='N',
        type=int,
        default=None,
        help='(optional) run number; if provided, we switch to offline mode'
    )

    args = parser.parse_args()

    exp     = args.exp
    run_num = args.run

    #print(f"exp: {exp!r}, runnum: {run_num!r}")

    mode = 'online' if run_num is None else 'offline'

    return mode, exp, run_num

def init(rank, mode, exp, run_num, config, callbacks):

    if mode == 'offline':
        h5_dir = config['h5']['path1'] + exp + config['h5']['path2']
        h5_path = h5_dir + config['h5']['name1'] + str(run_num) + config['h5']['name2']
        os.makedirs(h5_dir, exist_ok=True)

        log_dir = config['log']['path1'] + exp + config['log']['path2']
        os.makedirs(log_dir, exist_ok=True)

        pattern = h5_path[:-3]+'_*.h5'
        files_to_delete = glob.glob(pattern)+glob.glob(h5_path)    
        if rank==0:
            for file in files_to_delete:
                os.remove(file)
                print(f"Deleted {file}")

        if config.get('live', False):
            os.environ['PS_SMD_MAX_RETRIES'] = str(config.get('wait_time', 60))
        else:
            config['live'] = False

        
        if config['max_events'] is not None:
            ds = DataSource(exp=exp,run=run_num, live = config['live'], max_events=config['max_events'], monitor=True) 
        else:
            ds = DataSource(exp=exp,run=run_num, live = config['live'], monitor=True)             
        
        smd = ds.smalldata(filename=h5_path, batch_size=config['batch_size'])

    elif mode == 'online':
        # ds = DataSource(shmem='tmo_meb1')
        ###
        ds = DataSource(exp='tmo101333524',run=768)             
        #####
        smd = ds.smalldata(batch_size=1, callbacks=callbacks)        
    return ds, smd
  



def nsify(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: nsify(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [nsify(x) for x in d]
    else:
        return d

def read_config(fn,namespace=False):
    with open(fn, "r") as f:
        params = yaml.safe_load(f)
    if namespace: params = nsify(params)    
    return params


def dict_to_yaml_file(data: dict, filepath: str, *, sort_keys: bool = False) -> None:
    with open(filepath, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=sort_keys, default_flow_style=False)
