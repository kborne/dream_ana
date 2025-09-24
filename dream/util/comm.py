import numpy as np
from dream.util.histogram import worker_sparse_hist1d_fast, worker_sparse_hist2d_fast, group_sparse_hist1d_fast
from dream.util.misc import head_match
from dream.util.plots_comm import MultiLineWorkerPlot, RollAvgWorkerPlot, ScanVarWorkerPlot, Scan2VarWorkerPlot, Hist1DWorkerPlot, Hist2DWorkerPlot , ScanHist1DWorkerPlot

from dream.util.plots_comm import (SigBkg1DWorker, RollAvg1DFuncWorkerPlot, SingleLineFuncWorkerPlot, RollAvg1DWorkerPlot,
Hist1DFuncWorkerPlot, RollAvgFuncWorkerPlot, ScanVarFuncWorkerPlot, Scan2VarFuncWorkerPlot, SingleLineWorkerPlot, SingleImageWorkerPlot,
ScanHist1DFuncWorkerPlot, Hist1DFuncWorkerPlot, Hist2DFuncWorkerPlot)

# Map plot types to their corresponding handler classes
PLOT_CLASS_MAP = {
    'multiline': MultiLineWorkerPlot,
    'rollavg': RollAvgWorkerPlot,
    'scan_var': ScanVarWorkerPlot,
    'scan2_var': Scan2VarWorkerPlot,
    'scan_hist1d': ScanHist1DWorkerPlot,
    'hist1d': Hist1DWorkerPlot,
    'hist2d': Hist2DWorkerPlot,
    'sigbkg1d': SigBkg1DWorker,
    'rollavg1d': RollAvg1DWorkerPlot,
    'rollavg1d_func': RollAvg1DFuncWorkerPlot,
    'singleline': SingleLineWorkerPlot,
    'singleline_func': SingleLineFuncWorkerPlot,
    'singleimage': SingleImageWorkerPlot,
    'hist1d_func': Hist1DFuncWorkerPlot,
    'rollavgfunc': RollAvgFuncWorkerPlot,
    'scan_var_func': ScanVarFuncWorkerPlot,
    'scan2_var_func': Scan2VarFuncWorkerPlot,
    'scan_hist1d_func': ScanHist1DFuncWorkerPlot,
    'hist2d_func': Hist2DFuncWorkerPlot,
}

class comm_online:
    def __init__(self, config, requested_vars_by_detector):
        # Store parameters
        self.nacc1 = int(config['nacc'])
        self.handlers = []

        # Build data accumulator
        self.data_dict_acc = {
            f"{k1}:{k2}": np.zeros(0, float)
            for detector in requested_vars_by_detector.values()
            for k1, sub in detector.items()
            for k2 in sub
        }

        # Instantiate handlers based on config
        for name, p in config.get('plots', {}).items():
            plot_type = p.get('type')
            PlotClass = PLOT_CLASS_MAP.get(plot_type)
            if not PlotClass:
                raise ValueError(f"Unknown plot type '{plot_type}' for plot '{name}'")
            # Initialize the handler; each class is responsible for its own setup
            self.handlers.append(PlotClass(name, p))

    def histogram(self):
        self.data_dict = {}
        for h in self.handlers:
            h.accumulate(self.data_dict_acc, self.data_dict)
#        return self.data_dict


    def send(self, rank, smd, nevt, evt, evt_dict):       
    
        for k1 in evt_dict.keys():
            if head_match(k1.split('_')[0], ['wf', 'pdd', 'atm', 'fzp']):
                continue
            elif k1 == 'x':
                for k2 in evt_dict[k1].keys():
                    self.data_dict_acc[k2] = np.append(self.data_dict_acc[k2], evt_dict[k1][k2])                
            else:
                for k2 in evt_dict[k1].keys():
                    self.data_dict_acc[k1+':'+k2] = np.concatenate([self.data_dict_acc[k1+':'+k2], evt_dict[k1][k2]])
 
    
        if nevt%self.nacc1==0:       
                
            for k1 in evt_dict.keys():
                if head_match(k1.split('_')[0], ['wf', 'pdd', 'atm', 'fzp']):
                    for k2 in evt_dict[k1].keys():
                        self.data_dict_acc[k1+':'+k2] = evt_dict[k1][k2]
                                                                                                      
               
            self.histogram()
            self.data_dict['rank'] = rank
            smd.event(evt, self.data_dict)
    
            for k in self.data_dict_acc.keys():
                self.data_dict_acc[k] = np.zeros(0, dtype=float)





class comm_offline:
    def __init__(self,
                 config):
        
        self.config = config
        
    def send(self, rank, smd, nevt, evt, evt_dict):
      
        data_dict = {}
        if 'uniform' in self.config['data'].keys():
            for k in self.config['data']['uniform'].keys():
                if 'uniform' not in data_dict.keys(): data_dict['uniform'] = {}
                data_dict['uniform'][k] = {}
                max_len = self.config['data']['uniform'][k]['len']   
                
                if 'fvar' in self.config['data']['uniform'][k].keys():
                    for var in self.config['data']['uniform'][k]['fvar']:
                      
                        data_dict['uniform'][k][var] = evt_dict[k][var]
                
                for var in self.config['data']['uniform'][k]['var']:
                    temp = np.full((max_len,), np.nan)
                    var_len = len(evt_dict[k][var])
                    if var_len>0:
                        temp_len = min(var_len, max_len) 
                        temp[:temp_len] = evt_dict[k][var][:temp_len]                           
                    data_dict['uniform'][k][var] = temp
                
        if 'ragged' in self.config['data'].keys():
            for k in self.config['data']['ragged'].keys():
                if 'ragged' not in data_dict.keys(): data_dict['ragged'] = {}
                data_dict['ragged']['var_'+k] = {}                
                for var in self.config['data']['ragged'][k]['var']:
                    data_dict['ragged']['var_'+k][var] = evt_dict[k][var]

                if 'x' in evt_dict.keys():
                    for xk in evt_dict['x'].keys():
                        data_dict['ragged']['var_'+k][xk] = np.full(evt_dict[k][var].shape, evt_dict['x'][xk])

        
        if 'ragged_split' in self.config['data'].keys():
            for k in self.config['data']['ragged_split'].keys():
                if 'ragged' not in data_dict.keys(): data_dict['ragged'] = {}
                data_dict['ragged'][k] = {}
                for var in self.config['data']['ragged_split'][k]['var']:
                    data_dict['ragged'][k]['var_'+var] = {var: evt_dict[k][var]}

                    if 'x' in evt_dict.keys():
                        for xk in evt_dict['x'].keys():
                            data_dict['ragged'][k]['var_'+var][xk] = np.full(evt_dict[k][var].shape, evt_dict['x'][xk], dtype=float)
              
        if 'x' in evt_dict.keys():
            data_dict['x'] = evt_dict['x']

        smd.event(evt, data_dict)

        if nevt%2000==0: print('rank:', rank, 'nevt:',nevt)





















