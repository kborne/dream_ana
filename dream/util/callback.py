from psmon import publish
from collections import deque
import numpy as np
from dream.util.plots_callback import (
    MultiLinePlot, Hist1DPlot, Hist2DPlot,
    RollAvgPlot, ScanVarPlot, Scan2VarPlot, ScanHist1DPlot, SingleImagePlot
)

from dream.util.plots_callback import (
SigBkg1DPlot, RollAvg1DPlot, SingleLinePlot, Hist1DFuncPlot
)

from collections import deque
import numpy as np

# Map config 'type' strings (including “func” variants) to their Plot classes.
# Func‐variants generally reuse the same class as their non‐func counterparts,
# except for hist1d_func which has its own Hist1DFuncPlot.
PLOT_CLASS_MAP = {
    # multiline
    'multiline':        MultiLinePlot,

    # rolling averages
    'rollavg':          RollAvgPlot,
    'rollavg_func':     RollAvgPlot,
    'rollavg1d':        RollAvg1DPlot,
    'rollavg1d_func':   RollAvg1DPlot,

    # single‐line and image
    'singleline':      SingleLinePlot,
    'singleline_func':  SingleLinePlot,
    'singleimage':     SingleImagePlot,

    # 1D histograms
    'hist1d':           Hist1DPlot,
    'hist1d_func':      Hist1DFuncPlot,    # uses its own Func variant

    # 2D histograms
    'hist2d':           Hist2DPlot,
    'hist2d_func':      Hist2DPlot,

    # scan means
    'scan_var':         ScanVarPlot,
    'scan_var_func':    ScanVarPlot,
    'scan2_var':        Scan2VarPlot,
    'scan2_var_func':   Scan2VarPlot,

    # scan histograms
    'scan_hist1d':      ScanHist1DPlot,
    'scan_hist1d_func': ScanHist1DPlot,

    # signal–background fits
    'sigbkg1d':         SigBkg1DPlot,
}

class callback_online:
    def __init__(self, rank, numworkers, config):
        # derive all bin‐edges & scan‐axes

        self.rank       = rank
        self.config     = config
        self.numworkers = numworkers
        self.numendrun  = 0
        self.numupdates  = 0
        self.nacc1 = int(config['nacc'])
        self.nacc2 = numworkers
        #if self.rank==0: print('nacc2:', self.nacc2)

        # instantiate all handlers in one loop
        self.handlers = []
        for name, p in config['plots'].items():
            plot_type = p.get('type')
            PlotClass = PLOT_CLASS_MAP.get(plot_type)
            if PlotClass is None:
                raise ValueError(f"Unknown plot type '{plot_type}' for plot '{name}'")
            handler = PlotClass(name, p)
            self.handlers.append(handler)



    def smalldata(self, data_dict):
        # end‐of‐run reset
        if 'endrun' in data_dict:
            self.numendrun += 1
            if self.numendrun == self.numworkers:
                self.numendrun = 0
                self.numupdates = 0
                for h in self.handlers:
                    h._reset()
            return

        # accumulate new event
        self.numupdates += 1
        for h in self.handlers:
            h._accumulate(data_dict)

        # publish every nacc2 events
        if self.numupdates % self.nacc2 == 0:
            num = self.numupdates*self.nacc1
            print('num:', num)
            publish.init()
            for h in self.handlers:
                h._publish(num)
