#!/usr/bin/env python
# coding: utf-8

#/sdf/group/lcls/ds/tools/conda_envs/dream/config/dream/alg.yaml
import sys
import argparse


import psana as ps
import numpy as np

from scipy.signal import find_peaks

import matplotlib.pyplot as plt 
from matplotlib import colors
plt.style.use('bmh')


from psana.hexanode.PyCFD import PyCFD


sys.path.append('/sdf/group/lcls/ds/tools/smalldata_tools/pedplot/summaries/')
import panel as pn
from summary_utils import prepareHtmlReport

parser = argparse.ArgumentParser(description='Preprocessing Tabulation Application')
parser.add_argument('-r', '--run', type=int, required=True, help='Run number')
parser.add_argument('-e', '--experiment', type=str,  required=True,help='experiment name e.g., tmo101247125')

# CFD_params = {'sample_interval': 0.16826923,
#               'polarity' : "Negative"
#               'delay' : 4,
#               'fraction': 1,
#               'threshold': 500,
#               'walk': 0,
#               'timerange_low': 600,
#               'timerange_high': 15_000.0,
#               'offset': 16384}

CFD_params = {'sample_interval': 0.16826923,
              'polarity' : "Positive",
              'delay' : 4,
              'fraction': 1,
              'threshold': 500/16,
              'walk': 0,
              'timerange_low': 600,
              'timerange_high': 15_000.0,
              'offset': 0}


CFD = PyCFD(CFD_params)




ARGS = parser.parse_args()

run = ARGS.run
exp= ARGS.experiment

print('experiment: ',exp)
print('run: ', run)

max_events = 10_000
ds = ps.DataSource(exp=exp, run=run,max_events=max_events)
myrun = next(ds.runs())

det_name = {'u':1,'v':1,'w':1,'mcp':0}

fex_len = int(myrun.Detector('dream_hsd_lmcp').raw._seg_configs()[0].config.user.fex.gate_ns*(59_400/10)*(1/1000)+20)


det_methods = {n : myrun.Detector('dream_hsd_l'+n) for n in det_name.keys()}
det_datas = {n : {i: np.nan*np.ones((max_events,fex_len)) for i in range(c+1)} for n,c in det_name.items()}


baselines = {n : 
             {i: float(myrun.Detector('dream_hsd_l'+n).raw._seg_configs()[c].config.user.fex.corr.baseline) 
              for i in range(c+1)} 
             for n,c in det_name.items()}



for nevt,evt in enumerate(myrun.events()):
    # print(nevt)

    for name,chans in det_name.items():
        # print('\t',name)

        if det_methods[name].raw._segments(evt) is None:
            continue
            
        x_n = det_methods[name].raw.padded(evt)

        if x_n is not None:

            for c in range(chans+1):
                # print('\t \t',c)
    
                det_datas[name][c][nevt] = -(x_n[c][0]-baselines[name][c])/16
                # det_datas[name][c][nevt] = x_n[c][0]

print('deleting nan')

for name,chans in det_name.items():
    print('\t',name)
    for c in range(chans+1):
        nan_indx= np.where(np.isnan(det_datas[name][c][:,0]))[0]
        det_datas[name][c] = np.delete(det_datas[name][c],nan_indx,axis=0)




p1 = plt.figure()
for n,cn in det_name.items():
    for c in range(cn+1):
        plt.hist(800/4096*np.max(det_datas[n][c][:,10_000:],axis=1),
                 np.arange(0,800,2),
                 histtype ='step',
                 label = f'{n}:{c}');
plt.xlabel('FEX maximum / mV')
plt.legend()
plt.yscale('log')
plt.title(f'run {run } \n FEX-maximum')


tabs = pn.Tabs(p1)
prepareHtmlReport(tabs,exp,run,f"FEX-maximum/{run}")




def peak_properties(spec_i,times):
    # pks,_ = find_peaks(spec_i,**hf_configs)
    pts_t = CFD.CFD(spec_i,times)
    pks = [np.searchsorted(times,q) for q in pts_t]

    # pks,_ = find_peaks(spec_i,**hf_configs)
    phs = spec_i[pks]
    pks_90 = np.zeros(len(pks),dtype=int)
    pks_10 = np.zeros(len(pks),dtype=int)
    for p,pk in enumerate(pks):
        val = spec_i[pk]
        i = 0
        while val>0.9*spec_i[pk]:
            val = spec_i[pk-i]
            i+=1
        pks_90[p]=pk-i
        i = 0
        while val>0.1*spec_i[pk]:
            val = spec_i[pk-i]
            i+=1
        pks_10[p]=pk-i
    rts = pks_90-pks_10
    return pts_t,phs,rts


time_axis = np.arange(fex_len)*10_000/59400

for n,cn in det_name.items():

    for c in range(cn+1):
        p1 = plt.figure()
        
        for i in range(4):
            
            plt.subplot(2,2,i+1)
            
            spec_i = det_datas[n][c][i]
            # pts_t = CFD.CFD(spec_i,time_axis)
            pts_t,phs,rts = peak_properties(spec_i,time_axis)
            pks = [np.searchsorted(time_axis,q) for q in pts_t]
            len(pks)
            plt.plot(time_axis,spec_i*800/4096)
            
            if len(pks)>0:
                plt.plot(time_axis[pks],spec_i[pks]*800/4096,'v')

            # plt.xlim([(pks[0]-500)*0.000168,(pks[0]+500)*0.000168])
            # plt.ylim([-100*800/4096,1000*800/4096])
        plt.suptitle(f'run {run} \n {n}:{c}')
        plt.tight_layout()
        tabs = pn.Tabs(p1)
        prepareHtmlReport(tabs,exp,run,f"/Hit-found sample/ Hit-found {n} ~ {c}/{run}")


det_hf = {n : {i: np.nan*np.ones((np.shape(det_datas[n][c])[0],50)) for i in range(c+1)} for n,c in det_name.items()}
det_ph = {n : {i: np.nan*np.ones((np.shape(det_datas[n][c])[0],50)) for i in range(c+1)} for n,c in det_name.items()}
det_rt = {n : {i: np.nan*np.ones((np.shape(det_datas[n][c])[0],50)) for i in range(c+1)} for n,c in det_name.items()}

for n,cn in det_name.items():
    print(n)
    for c in range(cn+1):
        print('\t'+str(c))
        for i,spec_i in enumerate(det_datas[n][c]):
            
            hf,ph,rt= peak_properties(spec_i,time_axis)
            if len(hf)<50:
                det_hf[n][c][i,0:len(hf)]=hf
                det_ph[n][c][i,0:len(hf)]=ph
                det_rt[n][c][i,0:len(hf)]=rt
            else:
                print('\t \t too many hits')



p1 = plt.figure()
for n,cn in det_name.items():
    for c in range(cn+1):
        plt.hist(np.sum(~np.isnan(det_hf[n][c]),axis=1),np.arange(0,50),
                 histtype ='step',label = f'{n}:{c}');
plt.xlabel('Hits / shot')
plt.legend()
plt.suptitle(f'Run {run} \n Hits-per-shot')



tabs = pn.Tabs(p1)

prepareHtmlReport(tabs,exp,run,f"/Hits-per-shot/{run}")


p1 = plt.figure()
for n,cn in det_name.items():
    for c in range(cn+1):
        plt.hist((np.ndarray.flatten(det_hf[n][c])),
                 bins = np.arange(0,20_000,1),
                 histtype ='step',label = f'{n}:{c}');
plt.xlabel('Timing / ns')
plt.legend()
plt.yscale('log')
plt.title(f'run {run} \n Timing spectra')
tabs = pn.Tabs(p1)

prepareHtmlReport(tabs,exp,run,f"/Timing  /{run}")


p1 = plt.figure()
for n,cn in det_name.items():
    for c in range(cn+1):
        plt.hist((np.ndarray.flatten(np.diff(det_hf[n][c],axis=1))),
                 bins = np.arange(0,10_000,10),
                 histtype ='step',label = f'{n}:{c}');
plt.xlabel('delta Timing / ns')
plt.legend()
plt.yscale('log')
plt.title(f'run {run} \n delta Timing')
tabs = pn.Tabs(p1)
prepareHtmlReport(tabs,exp,run,f"/delta Timing  /{run}")

p1 = plt.figure()
for n,cn in det_name.items():
    for c in range(cn+1):
        plt.hist((np.ndarray.flatten(det_rt[n][c])),
                 np.arange(0,200,1),
                 histtype ='step',label = f'{n}:{c}');
plt.xlabel('Rise time / ns')
plt.legend()
plt.title(f'run {run} \n Rise times')
tabs = pn.Tabs(p1)
prepareHtmlReport(tabs,exp,run,f"/delta Timing  /{run}")



p1 = plt.figure()
for n,cn in det_name.items():
    for c in range(cn+1):
        plt.hist(800/4095*(np.ndarray.flatten(det_ph[n][c])),
                 np.arange(0,250),
                 histtype ='step',label = f'{n}:{c}');
plt.xlabel('Peak height / mV')
plt.legend()
# plt.yscale('log')
plt.title(f'run {run} \n Peak Height')
tabs = pn.Tabs(p1)
prepareHtmlReport(tabs,exp,run,f"/Peak Height  /{run}")

p1 = plt.figure()
for i,n in enumerate(['u','v','w']):
    plt.subplot(2,2,i+1)
    plt.hist2d(np.sum(~np.isnan(det_hf[n][0]),axis=1),
               np.sum(~np.isnan(det_hf[n][1]),axis=1),
               bins = 2*[np.arange(0,30)],norm=colors.LogNorm())
    plt.xlim([0,20])
    plt.ylim([0,20])
    plt.axis('square')
    plt.title('wire '+n)
    plt.xlabel('0')
    plt.ylabel('1')
    plt.colorbar()
plt.suptitle(f'Run {run} \n Hits-per-shot correlation')
plt.tight_layout()
tabs = pn.Tabs(p1)
prepareHtmlReport(tabs,exp,run,f"/Hits-per-shot correlation  /{run}")


v_ts = []
v_td = []

test=0
for q1_n,q2_n,r_n in zip(*det_hf['u'].values(),det_hf['mcp'][0]):

    for q1_i in q1_n[~np.isnan(q1_n)]:
        for q2_i in q2_n[~np.isnan(q2_n)]:
            for r_i in r_n[~np.isnan(r_n)]:
                v_ts.append(q1_i+q2_i-2*r_i)
                v_td.append(q1_i-q2_i)

    test+=1
    if test>1000:
        break



det_ts = {n: [] for n in ['u','v','w']}
det_td = {n: [] for n in ['u','v','w']}


for n in ['u','v','w']:
    for q1_n,q2_n,r_n in zip(*det_hf[n].values(),det_hf['mcp'][0]):
    
        for q1_i in q1_n[~np.isnan(q1_n)]:
            for q2_i in q2_n[~np.isnan(q2_n)]:
                for r_i in r_n[~np.isnan(r_n)]:
                    det_ts[n].append(q1_i+q2_i-2*r_i)
                    det_td[n].append(q1_i-q2_i)
    

p1 = plt.figure()

ts_edges = np.arange(-100,100,0.25)
ts_bins = 0.5*(ts_edges[:-1]+ts_edges[1:])

for i,n in enumerate(['u','v','w']):
    # plt.subplot(2,2,i+1)
    cnts = np.histogram(np.array(det_ts[n]),ts_edges)[0]
    
    plt.plot(ts_bins-ts_bins[np.argmax(cnts)],
             cnts,
             label = n);
    plt.xlim([-25,25])
    plt.xlabel('Time Sums / ns')
    plt.legend()

plt.suptitle(f'Run {run} \n time sums')
tabs = pn.Tabs(p1)

prepareHtmlReport(tabs,exp,run,f"/Time sums  /{run}")


p1 = plt.figure()
for n in ['u','v','w']:
    plt.hist(0.168*np.array(det_td[n]),
             np.arange(-250,250,1),
             histtype='step',label = n);
plt.xlabel('Time Differences / ns')
plt.legend()

plt.suptitle(f'Run {run} \n time differences')
tabs = pn.Tabs(p1)
prepareHtmlReport(tabs,exp,run,f"/Time differences  /{run}")



p1 = plt.figure()    
for i,n in enumerate(['u','v','w']):
    
    det_tdn = np.array(det_td[n])
    td_bin = np.arange(-200,200,1)
    
    det_tsn = np.array(det_ts[n])
    ts_bin = np.arange(-10,10,0.25)
    
    cnts = np.argmax(np.histogram(np.array(det_tsn),ts_bins[:-1])[0])

    plt.subplot(2,2,i+1)
    plt.hist2d(det_tdn,det_tsn-ts_bins[cnts],
              [td_bin,ts_bin],
               norm=colors.LogNorm()
              );
    plt.title(n)
    plt.colorbar()
    plt.xlabel('Time Difference / ns')
    plt.ylabel('Time Sum / ns')
    tabs = pn.Tabs(p1)
plt.suptitle(f'Run {run} \n Time differences vs time sums')
plt.tight_layout()
prepareHtmlReport(tabs,exp,run,f"/Time differences vs time sums  /{run}")
