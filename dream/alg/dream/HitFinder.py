#The radius condition is currently removed.

import numpy as np


class HitFinder:

    def __init__(self, params):
                                #parameter initializations
        self.uRunTime = params['runtime_u']
        
        self.vRunTime = params['runtime_v']
        
        self.wRunTime = params['runtime_w']
        
        self.u_diff_offset = params['u_diff_offset']
        
        self.v_diff_offset = params['v_diff_offset']
        
        self.w_diff_offset = params['w_diff_offset']        
        
        
        self.uTSumAvg = params['tsum_avg_u']
        self.uTSumLow = self.uTSumAvg - params['tsum_hw_u']
        self.uTSumHigh = self.uTSumAvg + params['tsum_hw_u']
        
        self.vTSumAvg = params['tsum_avg_v'] 
        self.vTSumLow = self.vTSumAvg - params['tsum_hw_v']
        self.vTSumHigh = self.vTSumAvg + params['tsum_hw_v']     
          
        self.wTSumAvg = params['tsum_avg_w']
        self.wTSumLow = self.wTSumAvg - params['tsum_hw_w']
        self.wTSumHigh = self.wTSumAvg + params['tsum_hw_w']
       

        
        self.radius = params['rMCP']
        self.f_u = params['f_u']
        self.f_v = params['f_v']
        self.f_w = params['f_w']
        
         
        self.sqrt3 = np.sqrt(3.)
        
    
        
        
    def FindHits(self, McpSig, u1Sig, u2Sig, v1Sig, v2Sig, w1Sig, w2Sig):
        
        
        
        # t1u = (-self.uRunTime+2*McpSig+self.uTSumAvg+self.u_diff_offset)/2
        # t2u = (self.uRunTime+2*McpSig+self.uTSumAvg-self.u_diff_offset)/2
            
        # t1v = (-self.vRunTime+2*McpSig+self.vTSumAvg+self.v_diff_offset)/2
        # t2v = (self.vRunTime+2*McpSig+self.vTSumAvg-self.v_diff_offset)/2
            
        # t1w = (-self.wRunTime+2*McpSig+self.wTSumAvg+self.w_diff_offset)/2
        # t2w = (self.wRunTime+2*McpSig+self.wTSumAvg-self.w_diff_offset)/2   


        t1u = (-self.uRunTime+2*McpSig+self.uTSumAvg-np.abs(self.u_diff_offset))/2
        t2u = (self.uRunTime+2*McpSig+self.uTSumAvg+np.abs(self.u_diff_offset))/2
            
        t1v = (-self.vRunTime+2*McpSig+self.vTSumAvg-np.abs(self.v_diff_offset))/2
        t2v = (self.vRunTime+2*McpSig+self.vTSumAvg+np.abs(self.v_diff_offset))/2
            
        t1w = (-self.wRunTime+2*McpSig+self.wTSumAvg-np.abs(self.w_diff_offset))/2
        t2w = (self.wRunTime+2*McpSig+self.wTSumAvg+np.abs(self.w_diff_offset))/2           
        
        
        self.data_dict = {}
        
        self.data_dict['x'] = np.array([])
        self.data_dict['y'] = np.array([])
        self.data_dict['t'] = np.array([])


    
               
        for i_McpT, McpT in enumerate(McpSig):
           
            u1 = u1Sig[(u1Sig>t1u[i_McpT]) & (u1Sig<t2u[i_McpT])]
            u2 = u2Sig[(u2Sig>t1u[i_McpT]) & (u2Sig<t2u[i_McpT])]
            v1 = v1Sig[(v1Sig>t1v[i_McpT]) & (v1Sig<t2v[i_McpT])]
            v2 = v2Sig[(v2Sig>t1v[i_McpT]) & (v2Sig<t2v[i_McpT])]      
            w1 = w1Sig[(w1Sig>t1w[i_McpT]) & (w1Sig<t2w[i_McpT])]
            w2 = w2Sig[(w2Sig>t1w[i_McpT]) & (w2Sig<t2w[i_McpT])]                          
            
            u1u2_sum = u1[:,np.newaxis] + u2[np.newaxis,:] - 2*McpT 
            v1v2_sum = v1[:,np.newaxis] + v2[np.newaxis,:] - 2*McpT 
            w1w2_sum = w1[:,np.newaxis] + w2[np.newaxis,:] - 2*McpT 
            
            
            u1_ind, u2_ind = np.where((u1u2_sum>self.uTSumLow) & (u1u2_sum<self.uTSumHigh))
            v1_ind, v2_ind = np.where((v1v2_sum>self.vTSumLow) & (v1v2_sum<self.vTSumHigh))
            w1_ind, w2_ind = np.where((w1w2_sum>self.wTSumLow) & (w1w2_sum<self.wTSumHigh))
                    
            
            
            sub_u = u1[u1_ind]-u2[u2_ind]
            sub_v = v1[v1_ind]-v2[v2_ind]
            sub_w = w1[w1_ind]-w2[w2_ind]
            
            sub_uf = (sub_u-self.u_diff_offset)*self.f_u/2
            sub_vf = (sub_v-self.v_diff_offset)*self.f_v/2
            sub_wf = (sub_w-self.w_diff_offset)*self.f_w/2
                   
                    
            
            Xuv = sub_uf[:,np.newaxis] + 0*sub_vf[np.newaxis,:]
            Yuv = (sub_uf[:,np.newaxis] - 2*sub_vf[np.newaxis,:])/self.sqrt3     
            
            x = np.ravel(Xuv)
            y = np.ravel(Yuv)

            rMCP = np.sqrt(x**2+y**2)
            inds = rMCP<self.radius

            if inds.sum()>0:
                self.data_dict['x'] = np.concatenate([self.data_dict['x'],x[inds][0:1]],axis=0)
                self.data_dict['y'] = np.concatenate([self.data_dict['y'],y[inds][0:1]],axis=0)
                self.data_dict['t'] = np.concatenate([self.data_dict['t'],np.array([McpT])],axis=0)            
            
            # self.data_dict['x'] = np.concatenate([self.data_dict['x'],x[inds]],axis=0)
            # self.data_dict['y'] = np.concatenate([self.data_dict['y'],y[inds]],axis=0)
            # self.data_dict['t'] = np.concatenate([self.data_dict['t'],McpT*np.ones(int(inds.sum()))],axis=0)
            

            
            # Xuw = sub_uf[:,np.newaxis] + 0*sub_wf[np.newaxis,:]
            # Yuw = (2*sub_wf[np.newaxis,:] - sub_uf[:,np.newaxis])/self.sqrt3            
            # self.data_dict['x_uw'] = np.concatenate([self.data_dict['x_uw'],np.ravel(Xuw)],axis=0)
            # self.data_dict['y_uw'] = np.concatenate([self.data_dict['y_uw'],np.ravel(Yuw)],axis=0)   
            # self.data_dict['t_uw'] = np.concatenate([self.data_dict['t_uw'],McpT*np.ones(len(np.ravel(Xuw)))],axis=0)

            # Xvw = sub_vf[:,np.newaxis] + sub_wf[np.newaxis,:]
            # Yvw = (sub_wf[np.newaxis,:] - sub_vf[:,np.newaxis])/self.sqrt3                   
            # self.data_dict['x_vw'] = np.concatenate([self.data_dict['x_vw'],np.ravel(Xvw)],axis=0)
            # self.data_dict['y_vw'] = np.concatenate([self.data_dict['y_vw'],np.ravel(Yvw)],axis=0)   
            # self.data_dict['t_vw'] = np.concatenate([self.data_dict['t_vw'],McpT*np.ones(len(np.ravel(Xvw)))],axis=0)

        self.data_dict['n'] = np.array([len(self.data_dict['t'])])