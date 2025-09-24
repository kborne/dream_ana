import os
import numpy as np
from pathlib import Path
from dream.util.setup import read_config
from dream.alg.common.peak_finders import hsd_peak_finder
from dream.util.misc import lists_intersection
from .HitFinder import HitFinder

class dld_reconstructor:
    def __init__(self, det_id, requested_vars):

        self.det_id = det_id
        self.sign_z = 1. if self.det_id == 's' else -1.
        config_dir = os.getenv("CONFIGDIR")
        config_dir = config_dir + 'dream/'
        self.params = read_config(config_dir + 'alg.yaml')[self.det_id]

        self.sig_names = ['mcp', 'u1', 'u2', 'v1', 'v2', 'w1', 'w2']
        hsd_dict = self.params['det']['keys']
        self.mapping = {}

        for k1, vals in hsd_dict.items():
            layer = k1.split('_')[-1][1:]
            for j, k2 in enumerate(vals):
                sig_name = 'mcp' if layer=='mcp' else layer+str(j+1)
                self.mapping[k1+k2] = sig_name
                
        self.peak_finder = hsd_peak_finder(self.det_id, self.sig_names, self.mapping, self.params['det'], requested_vars)

        
        self.SHF = HitFinder(self.params['hr'])

        self.avail_vars_peak_finder = ['wf_'+self.det_id, 'tpks_'+self.det_id, 'len_tpks_'+self.det_id]

        if len(lists_intersection(self.avail_vars_peak_finder, requested_vars.keys())) > 0:
            self.requested_peak_finder_data = True
        else:
            self.requested_peak_finder_data = False
        

        
        self.k0 = 'hit_'+self.det_id
        self.avail_vars = ['n', 'z', 'y', 't']

        if self.k0 in requested_vars.keys():
            if len(lists_intersection(self.avail_vars, requested_vars[self.k0])) > 0:
                self.reconstruction = True
            else:
                self.reconstruction = False

        if self.reconstruction:
            self.requested = {}
            for a_var in self.avail_vars:
                if a_var in requested_vars[self.k0]:
                    self.requested[a_var] = True
                else:
                    self.requested[a_var] = False

        ###
        self.k_pp = 'ppc_'+self.det_id
        #self.avail_vars_pp = ['pp1', 'pp2']

        self.pipico = False
        if self.k_pp in requested_vars.keys():
            #if len(lists_intersection(self.avail_vars_pp, requested_vars[self.k_pp])) > 0:
            self.pipico = True
            self.reconstruction = True

                
        ###
        self.k_tp = 'tpc_'+self.det_id
        #self.avail_vars_tp = ['pt1', 'pt2', 'pt3']

        self.tripico = False
        if self.k_tp in requested_vars.keys():
            #if len(lists_intersection(self.avail_vars_tp, requested_vars[self.k_tp])) > 0:
            self.tripico = True      
            self.reconstruction = True
        
            
        self.data_dict = {}

    def __call__(self, *args, **kwargs):
        self.data_dict = {}
        self.reconstruct(*args, **kwargs)
        return self.data_dict
        
    def reconstruct(self, det, evt):
 
        self.peak_finder(det, evt)
        if self.requested_peak_finder_data: self.data_dict.update(self.peak_finder.data_dict)

        if self.reconstruction:    

            if self.peak_finder.peak_exist:
                ks = self.peak_finder.tpks_dict.keys()
                if len(ks) != 7:
                    for sig_name in self.sig_names:
                        if sig_name not in ks:
                            self.peak_finder.tpks_dict[sig_name] = np.array([])
                            self.peak_finder.len_tpks_dict[sig_name] = 0
                            
                self.SHF.FindHits(self.peak_finder.tpks_dict['mcp'],
                                  self.peak_finder.tpks_dict['u1'], self.peak_finder.tpks_dict['u2'],
                                  self.peak_finder.tpks_dict['v1'], self.peak_finder.tpks_dict['v2'],
                                  self.peak_finder.tpks_dict['w1'], self.peak_finder.tpks_dict['w2'])
                
                self.data_dict[self.k0] = {}
                for var in ['n', 't']:
                    if self.requested[var]:
                        self.data_dict[self.k0][var] = self.SHF.data_dict[var]

                if self.requested['z']: self.data_dict[self.k0]['z'] = self.sign_z*self.SHF.data_dict['y']
                if self.requested['y']: self.data_dict[self.k0]['y'] = self.SHF.data_dict['x']


                if self.pipico:
                    self.data_dict[self.k_pp] = {}
                    if self.SHF.data_dict['n'][0]>1:
                        partitioned = np.partition(self.SHF.data_dict['t'], 1)
                        pps = np.sort(partitioned[:2])
                        self.data_dict[self.k_pp]['pp1'] = np.array([pps[0]])
                        self.data_dict[self.k_pp]['pp2'] = np.array([pps[1]])

                    else:
                        for var in ['pp1', 'pp2']:
                            self.data_dict[self.k_pp][var] = np.array([])  
                

                if self.tripico:
                    self.data_dict[self.k_tp] = {}
                    if self.SHF.data_dict['n'][0]>2:
                        partitioned = np.partition(self.SHF.data_dict['t'], 2)
                        tps = np.sort(partitioned[:3])
                        self.data_dict[self.k_tp]['tp1'] = np.array([tps[0]])
                        self.data_dict[self.k_tp]['tp2'] = np.array([tps[1]])     
                        self.data_dict[self.k_tp]['tp3'] = np.array([tps[2]])   

                    else:
                        for var in ['tp1', 'tp2', 'tp3']:
                            self.data_dict[self.k_tp][var] = np.array([])               
           

                    
            else:
                self.data_dict[self.k0] = {}
                if self.requested['n']: self.data_dict[self.k0]['n'] = np.array([0])
                for var in ['z', 'y', 't']:
                    if self.requested[var]: self.data_dict[self.k0][var] = np.array([])                   

                if self.pipico:
                    self.data_dict[self.k_pp] = {}
                    for var in ['pp1', 'pp2']:
                        self.data_dict[self.k_pp][var] = np.array([])  

                if self.tripico:
                    self.data_dict[self.k_tp] = {}
                    for var in ['tp1', 'tp2', 'tp3']:
                        self.data_dict[self.k_tp][var] = np.array([])                     
                    
                      