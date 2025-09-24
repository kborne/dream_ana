from bisect import bisect_left
import numpy as np
from collections import deque
from psmon import publish
from psmon.plots import XYPlot, Image
from dream.util.histogram import gather_dense_hist1d_fast

class BasePlot:
    def __init__(self, name):
        self.name = name

    def _reset(self):
        raise NotImplementedError

    def _accumulate(self, data_dict):
        raise NotImplementedError

    def _publish(self, num_events):
        raise NotImplementedError


class MultiLinePlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        # config: list of variable‑keys
        self.vars = p['var']
        self._last_x = []
        self._last_y = []
        
    def _reset(self):
        self._last_x.clear()
        self._last_y.clear()

    def _accumulate(self, data_dict):
        x_arrays, y_arrays = [], []
        for var in self.vars:
            if var in data_dict:
                arr = np.asarray(data_dict[var])
                x_arrays.append(np.arange(arr.size))
                y_arrays.append(arr)
        self._last_x, self._last_y = x_arrays, y_arrays

    def _publish(self, num_events):
        if not self._last_y:
            return
        plot = XYPlot(
            num_events,
            self.name,
            self._last_x,
            self._last_y,
            formats=['-'] * len(self._last_y)
        )
        publish.send(self.name, plot)


class Hist1DPlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        # config: p['arange'] is { var_key: [start, stop, step] }
        start, stop, step = next(iter(p['arange'].values()))
        edges = np.arange(start, stop, step)
        self.centers = 0.5 * (edges[:-1] + edges[1:])
        self.dense   = np.zeros_like(self.centers, dtype=int)

    def _reset(self):
        self.dense.fill(0)

    def _accumulate(self, data_dict):
        key = self.name
        if key in data_dict:
            gather_dense_hist1d_fast(self.dense, data_dict[key])

    def _publish(self, num_events):
        plot = XYPlot(
            num_events,
            self.name,
            self.centers,
            self.dense,
            formats=['-']
        )
        publish.send(self.name, plot)


class Hist2DPlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        # config: p['arange'] has two entries, one per axis
        (x0, x1, dx), (y0, y1, dy) = p['arange'].values()
        xe = np.arange(x0, x1, dx)
        ye = np.arange(y0, y1, dy)
        self.dense = np.zeros((xe.size - 1, ye.size - 1), dtype=int)

    def _reset(self):
        self.dense.fill(0)

    def _accumulate(self, data_dict):
        key = self.name
        if key in data_dict:
            from dream.util.histogram import gather_dense_hist2d_fast
            gather_dense_hist2d_fast(self.dense, data_dict[key])

    def _publish(self, num_events):
        img = Image(
            num_events,
            self.name,
            np.log10(np.rot90(self.dense + 1e-5))
        )
        publish.send(self.name, img)


class RollAvgPlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        # config: p['window'] = {'w1':…, 'w2':…}
        w1 = p['window']['w1']
        w2 = p['window']['w2']
        self.window  = deque(maxlen=w1)
        self.history = deque(maxlen=w2)
        
    def _reset(self):
        self.window.clear()
        self.history.clear()

    def _accumulate(self, data_dict):
        key = self.name
        if key in data_dict:
            self.window.append(data_dict[key])

    def _publish(self, num_events):
        if not self.window:
            return
        avg = np.mean(self.window)
        self.history.append(avg)
        plot = XYPlot(
            num_events,
            self.name,
            xdata=np.arange(len(self.history)),
            ydata=list(self.history),
            formats=['-']
        )
        publish.send(self.name, plot)


class ScanVarPlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        # p may contain var/scan/decimals/norm but gatherer ignores them
        self.keys    = []
        self.idx_map = {}
        self.sums    = np.zeros((0,), float)
        self.counts  = np.zeros((0,), float)
        
    def _reset(self):
        self.keys.clear()
        self.idx_map.clear()
        self.sums = np.zeros((0,), float)
        self.counts = np.zeros((0,), float)

    def _accumulate(self, data_dict):
        key = self.name
        if key not in data_dict:
            return
        sorted_local, sums_local, counts_local = data_dict[key]
        for v in sorted_local:
            if v not in self.idx_map:
                pos = bisect_left(self.keys, v)
                self.keys.insert(pos, v)
                self.sums = np.insert(self.sums, pos, 0.0)
                self.counts = np.insert(self.counts, pos, 0.0)
                self.idx_map = {k: i for i, k in enumerate(self.keys)}
        idxs = np.array([self.idx_map[v] for v in sorted_local], dtype=int)
        self.sums[idxs] += sums_local
        self.counts[idxs] += counts_local

    def calc(self):
        if self.sums.size == 0:
            return None, None

        means = self.sums / self.counts
            
        return self.keys, means   
        
    def _publish(self, num_events):
        if self.sums.size == 0:
            return
        means = self.sums / self.counts
        plot = XYPlot(
            num_events,
            self.name,
            self.keys,
            means
        )
        publish.send(self.name, plot)


class Scan2VarPlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        self.keys1    = []
        self.keys2    = []
        self.idx1_map = {}
        self.idx2_map = {}
        self.sums     = np.zeros((0, 0), float)
        self.counts   = np.zeros((0, 0), float)

    def _reset(self):
        self.keys1.clear()
        self.keys2.clear()
        self.idx1_map.clear()
        self.idx2_map.clear()
        self.sums = np.zeros((0, 0), float)
        self.counts = np.zeros((0, 0), float)

    def _accumulate(self, data_dict):
        key = self.name
        if key not in data_dict:
            return
        k1, k2, sums_local, counts_local = data_dict[key]
        for v in np.unique(k1):
            if v not in self.idx1_map:
                pos = bisect_left(self.keys1, v)
                self.keys1.insert(pos, v)
                self.sums = np.insert(self.sums, pos, 0.0, axis=0)
                self.counts = np.insert(self.counts, pos, 0.0, axis=0)
                self.idx1_map = {k: i for i, k in enumerate(self.keys1)}
        for v in np.unique(k2):
            if v not in self.idx2_map:
                pos = bisect_left(self.keys2, v)
                self.keys2.insert(pos, v)
                self.sums = np.insert(self.sums, pos, 0.0, axis=1)
                self.counts = np.insert(self.counts, pos, 0.0, axis=1)
                self.idx2_map = {k: j for j, k in enumerate(self.keys2)}
        G1, G2 = self.sums.shape
        rows = np.repeat([self.idx1_map[v] for v in k1], len(k2))
        cols = np.tile([self.idx2_map[v] for v in k2], len(k1))
        np.add.at(self.sums, (rows, cols), sums_local.ravel())
        np.add.at(self.counts, (rows, cols), counts_local.ravel())

    def _publish(self, num_events):
        if self.sums.size == 0:
            return
        mean_mat = self.sums / self.counts
        img = Image(num_events, self.name, mean_mat)
        publish.send(self.name, img)


class ScanHist1DPlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        # config: p['arange'] = { scan_var: [start, stop, step] }
        start, stop, step = p['arange_var']#next(iter(p['arange_var'].values()))
        # compute number of bins
        bin_count = int((stop - start) / step)
        self.bin_count = bin_count
        self.keys      = []
        self.idx_map   = {}
        self.matrix    = np.zeros((0, bin_count), dtype=int)
        self.counts    = np.zeros((0,), dtype=float)
        
    def _reset(self):
        self.keys.clear()
        self.idx_map.clear()
        self.matrix = np.zeros((0, self.bin_count), dtype=int)
        self.counts = np.zeros((0,), dtype=float)

    def _accumulate(self, data_dict):
        key = self.name
        if key not in data_dict:
            return
        H_sp, keys_local, num_arr = data_dict[key]
        for v in keys_local:
            if v not in self.idx_map:
                pos = bisect_left(self.keys, v)
                self.keys.insert(pos, v)
                self.matrix = np.insert(self.matrix, pos, 0, axis=0)
                self.counts = np.insert(self.counts, pos, 0.0)
                self.idx_map = {k: i for i, k in enumerate(self.keys)}
        rows = np.array([self.idx_map[keys_local[r]] for r in H_sp.row], dtype=int)
        self.matrix[rows, H_sp.col] += H_sp.data
        for i, v in enumerate(keys_local):
            self.counts[self.idx_map[v]] += num_arr[i]

    def _publish(self, num_events):
        if self.matrix.size == 0:
            return
        scan_normed = self.matrix / self.counts[:, None]
        img = Image(num_events, self.name, scan_normed)
        publish.send(self.name, img)


class SingleLinePlot(BasePlot):

    def __init__(self, name, p):
        super().__init__(name)
        self.centers = None
        self.dense   = None

    def _reset(self):# -> None:
        self.centers= None
        self.dense = None

    def _accumulate(self, data_dict: dict):# -> None:
        if self.name in data_dict:
            arr = np.asarray(data_dict[self.name])
            if self.centers is None: self.centers= np.arange(arr.size)
            self.dense = arr

    def calc(self):
        if self.dense is None or self.dense.size == 0:
            return None, None
            
        return self.centers, self.dense    
    
    def _publish(self, num_events: int):# -> None:
        if self.dense is None or self.dense.size == 0:
            return
        plot = XYPlot(
            num_events,
            self.name,
            self.centers,
            self.dense,
            formats=['-']
        )
        publish.send(self.name, plot)

class SingleImagePlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        self.dense = None
        
    def _reset(self):# -> None:
        self.dense = None

    def _accumulate(self, data_dict: dict):# -> None:
        if self.name in data_dict:
            # no checks, assume caller passed a valid 2D array
            self.dense = data_dict[self.name]

    def _publish(self, num_events: int):# -> None:
        if self.dense is None:
            return
        img = Image(
            num_events,
            self.name,
            self.dense
        )
        publish.send(self.name, img)



class SigBkg1DPlot(BasePlot):
    """
    Gatherer for signal vs. background 1D outputs from SigBkg1DWorker.
    Delegates to different Plot classes for sig vs. bkg:
      – for 'singleline_func' uses SingleLineFuncPlot (sig) and RollAvg1DFuncPlot (bkg)
      – for 'hist1d_func' uses Hist1DFuncPlot for both
      – otherwise strips '_func' and looks up the class in PLOT_CLASS_MAP
    Only window, norm_type, and arange_var are passed through.
    """
    def __init__(self, name: str, p: dict):
        super().__init__(name)
        self.op_type  = p.get('op_type')
        plot_type     = p['plot_type']

        # --- special case: singleline_func uses two different classes ---
        if plot_type == 'singleline_func':
            self.plot_sig = SingleLinePlot(f"{name}_sig", p)
            self.plot_bkg = RollAvg1DPlot(f"{name}_bkg", p)
            self.numevents = 0
            return

        # --- special case: hist1d_func has its own Func class ---
        if plot_type == 'hist1d_func':
            cls = Hist1DFuncPlot

        # --- default: func variants share non‑func class ---
        else:
            base = plot_type.replace('_func', '')
            cls = PLOT_CLASS_MAP[base]

        # collect only the config keys each class needs
        common_kwargs = {}
        if 'window'     in p: common_kwargs['window']     = p['window']
        if 'norm_type'  in p: common_kwargs['norm_type']  = p['norm_type']
        if 'arange_var' in p: common_kwargs['arange_var'] = p['arange_var']

        # instantiate both sig & bkg with the same class & kwargs
        self.plot_sig = cls(f"{name}_sig", common_kwargs)
        self.plot_bkg = cls(f"{name}_bkg", common_kwargs.copy())
        self.numevents = 0


    def _reset(self):# -> None:
        self.plot_sig._reset()
        self.plot_bkg._reset()

    def _accumulate(self, data_dict: dict):# -> None:
        # pass through to sub-plots
        self.plot_sig._accumulate(data_dict)
        self.plot_bkg._accumulate(data_dict)

    def _publish(self, num_events: int):# -> None:
        # let sub-plots prepare their plots
        self.numevents = num_events
        # publish individual lines into cached arrays
        # extract arrays
        x_sig, sig = self.plot_sig.calc()
        x_bkg, bkg = self.plot_bkg.calc()
        # if either missing, skip
        if sig is None or bkg is None:
            return
        # build series
        x_arrays = [x_sig, x_bkg]
        y_arrays = [sig, bkg]
        formats  = ['-','-']
        labels   = [f"{self.name}_sig", f"{self.name}_bkg"]
        
        if self.op_type == 'div':
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = sig / bkg
            y_arrays.append(ratio)
            x_arrays.append(x_sig)
            formats.append('--')
            labels.append(f"sig/bkg")        
        else:
            y_arrays.append(sig - bkg)
            x_arrays.append(x_sig)
            formats.append('--')
            labels.append(f"sig-bkg")

        plot = XYPlot(
            num_events,
            self.name,
            x_arrays,
            y_arrays,
            formats=formats,
            leg_label = labels
        )
        publish.send(self.name, plot)
        

class RollAvg1DPlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        # config: p['window'] = integer
        window = p['window']
        self.window  = window
        self.buffer  = deque(maxlen=window)
        self.centers = None

    def _reset(self):# -> None:
        self.buffer.clear()

    def _accumulate(self, data_dict: dict):# -> None:
    
        if self.name not in data_dict:
            return
        arr = np.asarray(data_dict[self.name])
        if self.centers is None:
            # first time: set up x‑axis
            self.centers = np.arange(arr.size)

        if len(arr)>0: self.buffer.append(arr)

    def calc(self):
        if not self.buffer:
            return None, None
        # stack and compute mean along axis 0
        stacked   = np.stack(self.buffer, axis=0)
        mean_arr  = stacked.mean(axis=0)
            
        return self.centers, mean_arr
    
    def _publish(self, num_events: int):# -> None:
        if not self.buffer:
            return
        # stack and compute mean along axis 0
     
        stacked   = np.stack(self.buffer, axis=0)
        mean_arr  = stacked.mean(axis=0)
        plot = XYPlot(
            num_events,
            self.name,
            self.centers,
            mean_arr,
            formats=['-']
        )
        
        publish.send(self.name, plot)
        


class Hist1DFuncPlot(BasePlot):
    def __init__(self, name, p):
        super().__init__(name)
        # config: p['centers'] or p['arange_var']+p['norm_type']
        if 'centers' in p:
            self.centers = np.asarray(p['centers'])
        else:
            start, stop, step = p.get('arange_var', p.get('arange')) 
            edges = np.arange(start, stop, step)
            self.centers = 0.5 * (edges[:-1] + edges[1:])
        self.dense    = np.zeros_like(self.centers, dtype=float)
        self.norm_sum = 0.0
        self.norm_type = p.get('norm_type')

    def _reset(self):# -> None:
        self.dense[:]   = 0.0
        self.norm_sum   = 0.0

    def _accumulate(self, data_dict: dict):# -> None:
        key_h = f'h1_{self.name}'
        if key_h in data_dict:
            gather_dense_hist1d_fast(self.dense, data_dict[key_h])
        if self.norm_type:
            key_n = f'norm_{self.name}'
            if key_n in data_dict:
                self.norm_sum += data_dict[key_n]

    def calc(self):
        if self.norm_type and self.norm_sum > 0:
            plot_data = self.dense / self.norm_sum
        else:
            plot_data = self.dense
            
        return self.centers, plot_data

    def _publish(self, num_events: int):# -> None:
        if self.norm_type and self.norm_sum > 0:
            plot_data = self.dense / self.norm_sum
        else:
            plot_data = self.dense
        plot = XYPlot(
            num_events,
            self.name,
            self.centers,
            plot_data,
            formats=['-']
        )
        publish.send(self.name, plot)
