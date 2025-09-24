import os
import numpy as np
from dream.util.setup import read_config
from scipy.ndimage import gaussian_filter1d

class scan:
    def __init__(self, requested_vars):

        self.params = {'det':{'keys':[]}}
        self.requested_vars = requested_vars
        self.data_dict = {}
        self.det_id = 'scan'

    def get_det_keys(self, run):
        try:
            self.scan_items = list(run.scaninfo.items())
        except Exception as err:
            print(err)
            self.scan_items = None
            
        
        for requested_var in self.requested_vars[self.det_id]:
            try:
                if requested_var == 'var1':
                    self.params['det']['keys'].append(self.scan_items[2][0][0])
                elif requested_var == 'var2':
                    self.params['det']['keys'].append(self.scan_items[3][0][0])       
            except Exception as err:
                self.params['det']['keys'].append(None)
            
    def __call__(self, *args, **kwargs):
        self.data_dict = {}  
        try:
            self.get_vars(*args, **kwargs)
        except Exception as err:
            self.data_dict['x'] = {}
            for requested_var in self.requested_vars[self.det_id]:
                self.data_dict['x'][self.det_id+':'+requested_var] = np.nan
             
        return self.data_dict

    def get_vars(self, det, evt):
        self.data_dict['x'] = {}
        for i, requested_var in enumerate(self.requested_vars[self.det_id]): 
            self.data_dict['x'][self.det_id+':'+requested_var] = det[self.params['det']['keys'][i]](evt) if det[self.params['det']['keys'][i]] is not None else np.nan   

class bld:
    def __init__(self, requested_vars):

        self.params = {'det':{'keys':[]}}
        self.requested_vars = requested_vars
        self.data_dict = {}
        self.det_id = 'bld'

    def get_det_keys(self, run):
        try:
            keys = run.detnames            
            for requested_var in self.requested_vars[self.det_id]:
                if requested_var not in keys: requested_var = None
                self.params['det']['keys'].append(requested_var)
        except Exception as err:
            print(err)
            for requested_var in self.requested_vars[self.det_id]:
                self.params['det']['keys'].append(None)
      
    def __call__(self, *args, **kwargs):
        self.data_dict = {}       
        try:
            self.get_vars(*args, **kwargs)
        except Exception as err:
            self.data_dict['x'] = {}
            for requested_var in self.requested_vars[self.det_id]:
                self.data_dict['x'][self.det_id+':'+requested_var] = np.nan           
               
        return self.data_dict

    def get_vars(self, det, evt):
        self.data_dict['x'] = {}
        for i, requested_var in enumerate(self.requested_vars[self.det_id]): 
            self.data_dict['x'][self.det_id+':'+requested_var] = det[self.params['det']['keys'][i]].raw.milliJoulesPerPulse(evt) if det[self.params['det']['keys'][i]] is not None else np.nan   
            if not self.data_dict['x'][self.det_id+':'+requested_var]:
                self.data_dict['x'][self.det_id+':'+requested_var] = np.nan 

class epics:
    def __init__(self, requested_vars):

        self.params = {'det':{'keys':[]}}
        self.requested_vars = requested_vars
        self.data_dict = {}
        self.det_id = 'epics'

    def get_det_keys(self, run):
        try:
            tuples = list(run.epicsinfo.keys())
            keys, _ = zip(*tuples)
            keys = list(keys)        
            for requested_var in self.requested_vars[self.det_id]:
                if requested_var not in keys: requested_var = None
                self.params['det']['keys'].append(requested_var)
        except Exception as err:
            print(err)    
            for requested_var in self.requested_vars[self.det_id]:
                self.params['det']['keys'].append(None)
                      
          
    def __call__(self, *args, **kwargs):
        self.data_dict = {}       
        try:
            self.get_vars(*args, **kwargs)
        except Exception as err:
            self.data_dict['x'] = {}
            for requested_var in self.requested_vars[self.det_id]:
                self.data_dict['x'][self.det_id+':'+requested_var] = np.nan           
                
        return self.data_dict

    def get_vars(self, det, evt):
        self.data_dict['x'] = {}
        for i, requested_var in enumerate(self.requested_vars[self.det_id]): 
            self.data_dict['x'][self.det_id+':'+requested_var] = det[self.params['det']['keys'][i]](evt) if det[self.params['det']['keys'][i]] is not None else np.nan   


class timing:
    def __init__(self, requested_vars):

        self.params = {'det':{'keys':[]}}
        self.requested_vars = requested_vars
        self.data_dict = {}
        self.det_id = 'timing'

    def get_det_keys(self, run):

        try:
            keys = run.detnames     
            var_key = 'timing'
            if var_key not in keys: var_key = None
            self.params['det']['keys'].append(var_key)            
        except Exception as err:
            print(err)
            self.params['det']['keys'].append(None)    
      
    def __call__(self, *args, **kwargs):
        self.data_dict = {}       
        try:
            self.get_vars(*args, **kwargs)
        except Exception as err:
            self.data_dict['x'] = {}
            for requested_var in self.requested_vars[self.det_id]:
                self.data_dict['x'][self.det_id+':'+requested_var] = np.nan 
                
        return self.data_dict

    def get_vars(self, det, evt):
        self.data_dict['x'] = {}
        det_timing = next(iter(det.values()))
        for i, requested_var in enumerate(self.requested_vars[self.det_id]): 
            if requested_var=='dest':
                self.data_dict['x'][self.det_id+':'+requested_var] = det_timing.raw.destination(evt) if det_timing is not None else np.nan      
            else:
                self.data_dict['x'][self.det_id+':'+requested_var] = det_timing.raw.eventcodes(evt)[int(requested_var)] if det_timing is not None else np.nan 


class atm:
    def __init__(self, requested_vars):
        
        self.det_id = 'atm'

        config_dir = os.getenv("CONFIGDIR")
        instrument = read_config(config_dir+'instrument.yaml')['instrument']
        config_dir = config_dir + instrument + '/'
        params = read_config(config_dir + 'alg.yaml')[self.det_id]
        self.params = params
        
        #self.requested_vars = requested_vars
        self.data_dict = {}
       
      
    def __call__(self, *args, **kwargs):
        self.data_dict = {}       
        try:
            self.get_vars(*args, **kwargs)
        except Exception as err:
            print('atm error:',err)
            self.data_dict['atm'] = {'line':[]}
            self.data_dict['atm'] = {'gline':[]}
                
        return self.data_dict

    def get_vars(self, det, evt):
        self.data_dict['atm'] = {}
        line = next(iter(det.values())).raw.raw(evt)
        self.data_dict['atm']['line'] = line
        self.data_dict['atm']['gline'] = gaussian_filter1d(line,self.params['gfw'])

       


class fzp:
    def __init__(self, requested_vars):
        self.det_id = 'fzp'

        config_dir = os.getenv("CONFIGDIR")
        instrument = read_config(config_dir+'instrument.yaml')['instrument']
        config_dir = config_dir + instrument + '/'
        params = read_config(config_dir + 'alg.yaml')[self.det_id]
        self.params = params
        self.hw_fzp = params['hw']
        self.requested_vars = requested_vars
        self.data_dict = {}
       
      
    def __call__(self, *args, **kwargs):
        self.data_dict = {}       
        try:
            self.get_vars(*args, **kwargs)
        except Exception as err:
            self.data_dict['x'] = {}
            for requested_var in self.requested_vars[self.det_id]:
                self.data_dict['x'][f"{self.det_id}:{requested_var}"] = np.nan
                
        return self.data_dict

    def get_vars(self, det, evt):
        self.data_dict['x'] = {}
        prj = next(iter(det.values())).raw.raw(evt)
        xmax,m1,m2,area = self.PhotonSpectrumMoments(prj,self.hw_fzp)
        for name, value in zip(('xmax', 'm1', 'm2', 'area'), (xmax, m1, m2, area)):
            if name in self.requested_vars[self.det_id]:
                self.data_dict['x'][f"{self.det_id}:{name}"] = value

    
    def PhotonSpectrumMoments(self,proj,hw):
        """Get the center, FWHM, AOC of the photon spectrum by direct calculation of first, second moments and array sum.
        Parameters
        ----------
        proj: photon spectrum, 1D array
        hw: predefined half width of the region centerd around spectrum peak
        Returns
        -------
        center, width and area of the photon spectrum
    
        """
        x_max = proj.argmax()
    
        inda = max(0,x_max-hw)
        indb = min(x_max+hw,len(proj)-1)
        #ym = np.mean(proj[0:min(50,inda)])
        ym = np.min(proj)
    
        x = np.arange(inda,indb)
        y = proj[inda:indb]-ym
        y[y<0] = 0
    
        m1 = np.average(x,weights=y)
        m2 = np.sqrt(np.average((x-m1)**2,weights=y))
    
        return x_max,m1,m2,np.sum(y).astype(float)
    
