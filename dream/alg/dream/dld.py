import os
import numpy as np
from pathlib import Path
from dream.util.setup import read_config
from dream.alg.common.peak_finders import hsd_peak_finder
#from dream.alg.common.peak_finders_scipy import hsd_peak_finder
from dream.lib.libASort import PyASort
from dream.util.misc import lists_intersection
from itertools import combinations

class dld_reconstructor:
    def __init__(self, det_id, requested_vars, rank, **kwargs):

        self.det_id = det_id
        self.sign_z = 1. if self.det_id == 's' else -1.
        config_dir = os.getenv("CONFIGDIR")
        config_dir = config_dir + 'dream/'
        self.params = read_config(config_dir + 'alg.yaml')[self.det_id]

        if rank==0:
            print('DET ID: ', self.det_id)
            print('ALG: ', 'dld')
            print('CONFIG:')
            print(self.params)  

        self.sig_names = ['mcp', 'u1', 'u2', 'v1', 'v2', 'w1', 'w2']
        hsd_dict = self.params['det']['keys']
        self.mapping = {}

        for k1, vals in hsd_dict.items():
            layer = k1.split('_')[-1][1:]
            for j, k2 in enumerate(vals):
                sig_name = 'mcp' if layer=='mcp' else layer+str(j+1)
                self.mapping[k1+k2] = sig_name
                
        self.peak_finder = hsd_peak_finder(self.det_id, self.sig_names, self.mapping, self.params['det'], requested_vars)

        setting_names = ['pos_offset_x', 'pos_offset_y', 'tsum_hw_u', 'tsum_hw_v', 'tsum_hw_w', 'f_u', 'f_v', 'f_w', 'w_offset', 'runtime_u', 'runtime_v', 'runtime_w', 'rMCP', 'dtime_dld', 'dtime_mcp', 'mth_max']
        settings = [self.params['hr'][setting_name] for setting_name in setting_names]
        self.RHF = PyASort()
        s_corr, p_corr = 1, 1
        _ = self.RHF.init_sorter(config_dir, 0, 1, s_corr, p_corr, *settings)
    

        self.sig_offset_dict = {}
        for sig_name in self.sig_names[1:]:
            if sig_name[1] == '1':
                self.sig_offset_dict[sig_name] = (self.params['hr']['tsum_avg_'+sig_name[0]] + self.params['hr'][sig_name[0]+'_diff_offset'])/2
            elif sig_name[1] == '2':
                self.sig_offset_dict[sig_name] = (self.params['hr']['tsum_avg_'+sig_name[0]] - self.params['hr'][sig_name[0]+'_diff_offset'])/2  
        self.sig_offset_dict['mcp'] = 0.

        self.avail_vars_peak_finder = ['wf_'+self.det_id, 'pdd_'+self.det_id, 'tpks_'+self.det_id, 'len_tpks_'+self.det_id]

        if len(lists_intersection(self.avail_vars_peak_finder, requested_vars.keys())) > 0:
            self.requested_peak_finder_data = True
        else:
            self.requested_peak_finder_data = False
        

        ###
        self.k0 = 'hit_'+self.det_id
        self.k_diag = 'diag_'+self.det_id
        self.avail_vars_k0 = ['n', 'z', 'y', 't', 'm']
        self.avail_vars_k_diag = ['pos_u', 'tsum_u', 'pos_v', 'tsum_v', 'pos_w', 'tsum_w']

        self.reconstruction_k0 = False
        if self.k0 in requested_vars.keys():
            if len(lists_intersection(self.avail_vars_k0, requested_vars[self.k0])) > 0:
                self.reconstruction_k0 = True

        
        self.requested = {}
        if self.reconstruction_k0:       
            for a_var in self.avail_vars_k0:
                if a_var in requested_vars[self.k0]:
                    self.requested[a_var] = True
                else:
                    self.requested[a_var] = False        

        self.reconstruction_k_diag = False
        if self.k_diag in requested_vars.keys():
            if len(lists_intersection(self.avail_vars_k_diag, requested_vars[self.k_diag])) > 0:
                self.reconstruction_k_diag = True       
                self.pos_sum_index = {'pos_u': 0, 'tsum_u': 1, 'pos_v': 2, 'tsum_v': 3, 'pos_w': 4, 'tsum_w': 5}

        self.reconstruction = (self.reconstruction_k0 or self.reconstruction_k_diag)
        # if self.reconstruction_k_diag:
        #     for a_var in self.avail_vars_k_diag:
        #         if a_var in requested_vars[self.k0]:
        #             self.requested[a_var] = True
        #         else:
        #             self.requested[a_var] = False
                    
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
        #self.avail_vars_tp = ['tp1', 'tp2', 'tp3']

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
        
    def reconstruct(self, det, evt, *args, **kwargs):

        self.peak_finder(det, evt)
        if self.requested_peak_finder_data: self.data_dict.update(self.peak_finder.data_dict)

        if self.reconstruction:

            len_peaks = 1
            if self.peak_finder.peak_exist:
                ks = self.peak_finder.tpks_dict.keys()
                if len(ks) != 7:
                    for sig_name in self.sig_names:
                        if sig_name not in ks:
                            self.peak_finder.tpks_dict[sig_name] = np.array([])
                            self.peak_finder.len_tpks_dict[sig_name] = 0
                            
                for sig_name in self.sig_names: 
                    len_peaks *= (len(self.peak_finder.tpks_dict[sig_name])+1)
                    self.RHF.set_peaks_arr(sig_name, 
                                           self.peak_finder.tpks_dict[sig_name] - self.sig_offset_dict[sig_name], 
                                           self.peak_finder.len_tpks_dict[sig_name])                                    
                

                if len_peaks > 170859375: #10000000:
                    self.data_dict[self.k0] = {}
                    if self.requested['n']: self.data_dict[self.k0]['n'] = np.array([0])
                    for var in ['z', 'y', 't', 'm']:
                        if self.requested[var]: self.data_dict[self.k0][var] = np.array([])  

                    if self.pipico:
                        self.data_dict[self.k_pp] = {}
                        for var in ['pp1', 'pp2']:
                            self.data_dict[self.k_pp][var] = np.array([])  

                    if self.tripico:
                        self.data_dict[self.k_tp] = {}
                        for var in ['tp1', 'tp2', 'tp3']:
                            self.data_dict[self.k_tp][var] = np.array([])         

                    if self.reconstruction_k_diag:
                        self.data_dict[self.k_diag] = {}
                        for k_pos_sum in self.pos_sum_index.keys():
                            self.data_dict[self.k_diag][k_pos_sum] = np.array([np.nan])                    
                            
                    return

                #print('marker 1')
                    
                self.RHF.pre_sort()  

                if self.reconstruction_k_diag:
                    self.data_dict[self.k_diag] = {}
                    pt_ready = self.RHF.pos_tsum_ready()
                    #print('marker 2')
                    if pt_ready:
                        pos_tsum = self.RHF.get_pos_tsum()
                        #print('marker 3')
                        for k_pos_sum in self.pos_sum_index.keys():
                            self.data_dict[self.k_diag][k_pos_sum] = np.array([pos_tsum[self.pos_sum_index[k_pos_sum]]])
                    else:
                        for k_pos_sum in self.pos_sum_index.keys():
                            self.data_dict[self.k_diag][k_pos_sum] = np.array([np.nan])                     
                        

                if self.reconstruction_k0:
                    self.data_dict[self.k0] = {}
                    self.RHF.sort()
                    #print('marker 4')
    
                    self.RHF.fill_hits()
                    #print('marker 5')
                    
                    hits_n = self.RHF.get_hits_n()
                    #print('marker 6')
                    if self.requested['n']: self.data_dict[self.k0]['n'] = np.array([hits_n])
                    if self.requested['z']: self.data_dict[self.k0]['z'] = self.sign_z*self.RHF.get_hits_y()
                    if self.requested['y']: self.data_dict[self.k0]['y'] = self.RHF.get_hits_x()
                    if self.requested['t']: self.data_dict[self.k0]['t'] = self.RHF.get_hits_t()
                    if self.requested['m']: self.data_dict[self.k0]['m'] = self.RHF.get_hits_method()


                    
                    
                if self.pipico:
                    self.data_dict[self.k_pp] = {}
                    if hits_n>1:
                        # partitioned = np.partition(self.RHF.get_hits_t(), 1)
                        # pps = np.sort(partitioned[:2])
                        # self.data_dict[self.k_pp]['pp1'] = np.array([pps[0]])
                        # self.data_dict[self.k_pp]['pp2'] = np.array([pps[1]]) 
                        
                        pairs = np.array(list(combinations(self.RHF.get_hits_t(), 2)))
                        self.data_dict[self.k_pp]['pp1'] = pairs[:,0]
                        self.data_dict[self.k_pp]['pp2'] = pairs[:,1]    
    
                
                    else:
                        for var in ['pp1', 'pp2']:
                            self.data_dict[self.k_pp][var] = np.array([])  
                

                if self.tripico:
                    self.data_dict[self.k_tp] = {}
                    if hits_n>2:
                        partitioned = np.partition(self.RHF.get_hits_t(), 2)
                        tps = np.sort(partitioned[:3])
                        self.data_dict[self.k_tp]['tp1'] = np.array([tps[0]])
                        self.data_dict[self.k_tp]['tp2'] = np.array([tps[1]])     
                        self.data_dict[self.k_tp]['tp3'] = np.array([tps[2]])   

                    else:
                        for var in ['tp1', 'tp2', 'tp3']:
                            self.data_dict[self.k_tp][var] = np.array([])              

            
            else:
                if self.reconstruction_k0:
                    self.data_dict[self.k0] = {}
                    if self.requested['n']: self.data_dict[self.k0]['n'] = np.array([0])
                    for var in ['z', 'y', 't', 'm']:
                        if self.requested[var]: self.data_dict[self.k0][var] = np.array([])  

                if self.pipico:
                    self.data_dict[self.k_pp] = {}
                    for var in ['pp1', 'pp2']:
                        self.data_dict[self.k_pp][var] = np.array([])  

                if self.tripico:
                    self.data_dict[self.k_tp] = {}
                    for var in ['tp1', 'tp2', 'tp3']:
                        self.data_dict[self.k_tp][var] = np.array([])    


                if self.reconstruction_k_diag:
                    self.data_dict[self.k_diag] = {}
                    for k_pos_sum in self.pos_sum_index.keys():
                        self.data_dict[self.k_diag][k_pos_sum] = np.array([np.nan])
     

                    
                      
