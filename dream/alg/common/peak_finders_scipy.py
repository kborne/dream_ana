import os
import numpy as np
from scipy.signal import find_peaks
from dream.util.setup import read_config

class hsd_peak_finder():
    def __init__(self, det_id, sig_names, mapping, params=None, requested_vars=None): 

        self.det_id = det_id
        self.mapping = mapping
        self.sig_names = sig_names
        if params is None:
            config_dir = os.getenv("CONFIGDIR")
            instrument = read_config(config_dir+'instrument.yaml')['instrument']
            config_dir = config_dir + instrument + '/'
            params = read_config(config_dir + 'alg.yaml')[self.det_id]['det']

        self.params = params
        self.finder = {}
        for k1, vals in self.params['keys'].items():
            for k2 in vals:
                self.finder[self.mapping[k1+k2]] = peak_finder_scipy(self.params['all'])

        self.ts_wf = None
        
        self.avail_vars = ['wf', 'fex', 'tpks', 'n_tpks']

        self.requested = {}
        self.data_dict = {}
        for k1 in self.avail_vars:
            self.requested[k1] = {}
            k1_p = k1+'_'+self.det_id
            if k1_p in requested_vars:
                self.data_dict[k1_p] = {}
                for k2 in sig_names:
                    if k2 in requested_vars[k1_p]:
                        self.requested[k1][k2] = True
                    else:
                        self.requested[k1][k2] = False
            else:
                for k2 in sig_names:
                    self.requested[k1][k2] = False       
    



    def __call__(self, *args, **kwargs):
    
        try:
            if self.params['fex']:
                self.find_peaks_fex(*args, **kwargs)  
            else:
                self.find_peaks_raw(*args, **kwargs)  
        except Exception as err:
            for k1 in self.avail_vars:
                k1_p = k1+'_'+self.det_id
                self.data_dict[k1_p] = {}
                for k2 in self.sig_names:     
                    self.data_dict[k1_p][k2] = np.array([])
                        
                
        return self.data_dict            

    def find_peaks_fex(self, det, evt):
        self.tpks_dict = {}
        self.n_tpks_dict = {}
        for k in self.data_dict.keys(): self.data_dict[k] = {}
        for k1 in self.params['keys'].keys():
            peaks = det[k1].raw.peaks(evt)
            for i, k2 in enumerate(peaks.keys()):
                key_pks = self.mapping[k1+str(k2)]
                starts = np.array(peaks[k2][0][0]).astype('float')
                amps = peaks[k2][0][1]
                tpks_all = np.empty((0,), dtype=float)
                for j, (start, amp) in enumerate(zip(starts, amps)):
                    amp = amp.astype('float')
                    ts = start + np.arange(len(amp))
                    
                    tpks = self.finder[key_pks](amp.astype(float),ts*0.1682692307692308)     
                     
                    if len(tpks)==0: continue

                    tpks_all = np.concatenate([tpks_all, tpks])

                
                self.tpks_dict[key_pks] = tpks_all
                self.n_tpks_dict[key_pks] = len(tpks_all)
                if self.requested['tpks'][key_pks]:
                    self.data_dict['tpks_'+self.det_id].update({key_pks: self.tpks_dict[key_pks]})

                if self.requested['n_tpks'][key_pks]:
                    self.data_dict['n_tpks_'+self.det_id].update({key_pks: self.n_tpks_dict[key_pks]})

        
    def find_peaks_raw(self, det, evt):
        self.tpks_dict = {}
        self.n_tpks_dict = {}    
        for k in self.data_dict.keys(): self.data_dict[k] = {}
        for k1 in self.params['keys'].keys():
            wfs = det[k1].raw.waveforms(evt)      
            for i, k2 in enumerate(wfs.keys()):
                key_pks = self.mapping[k1+str(k2)]
                if self.ts_wf is None:
                   self.ts_wf =  wfs[k2]['times']*1e9   

                if self.requested['wf'][key_pks]:
                    self.data_dict['wf_'+self.det_id].update({key_pks: wfs[k2][0].astype(float)})

                tpks_all = self.finder[key_pks](wfs[k2][0].astype(float), self.ts_wf)
            
                
                self.tpks_dict[key_pks] = tpks_all
                self.n_tpks_dict[key_pks] = len(tpks_all)   
                
                if self.requested['tpks'][key_pks]:
                    self.data_dict['tpks_'+self.det_id].update({key_pks: self.tpks_dict[key_pks]})

                if self.requested['n_tpks'][key_pks]:
                    self.data_dict['n_tpks_'+self.det_id].update({key_pks: self.n_tpks_dict[key_pks]})



class peak_finder_scipy:

    def __init__(self, params):
        self.prominence = params['prominence']
        self.threshold = params['threshold']
        self.polarity = 1. if params['polarity']=='Positive' else -1.
        self.timerange_low = params['timerange_low']
        self.timerange_high = params['timerange_high']
        self.offset = params['offset']


    def __call__(self, *args, **kwargs):
        return self.find_peaks(*args, **kwargs)

    def find_peaks(self, wf, wt):
        inds, _ = find_peaks(self.polarity*wf, prominence=self.prominence)
        return wt[inds]
