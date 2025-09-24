from dream.util.histogram import (
    worker_sparse_mean_sort,
    worker_sparse_mean_sort2d,
    worker_sparse_sort1d_fast,
    worker_sparse_hist1d_fast,
    worker_sparse_hist2d_fast
)

import numpy as np

from dream.util.misc import mk_func

class BaseWorkerPlot:
    def __init__(self, name):
        self.name = name
    def accumulate(self, data_acc, out_dict):
        raise NotImplementedError


class MultiLineWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        # list of variable names
        self.vars = p['var']
        # optional y-offset between lines
        self.offset = p.get('y_offset', 0)

    def accumulate(self, data_acc, out_dict):
        for i, var in enumerate(self.vars):
            if var in data_acc and len(data_acc[var]) > 0:                    
                arr = np.asarray(data_acc[var]) + i * self.offset
                out_dict[var] = arr


class RollAvgWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        self.var = p['var']


    def accumulate(self, data_acc, out_dict):
        if self.var in data_acc:
            arr = np.atleast_1d(data_acc[self.var])
            out_dict[self.name] = np.mean(arr)


class ScanVarWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        self.var = p['var']
        self.scan = p['scan']
        # allow 'decimal' or 'decimals'
        dec = p.get('decimals', p.get('decimal'))
        self.decimals = dec
        self.norm = p.get('norm')

    def accumulate(self, data_acc, out_dict):
        if self.var in data_acc and self.scan in data_acc:
            arr = np.atleast_1d(data_acc[self.var])
            arr_scan = np.atleast_1d(data_acc[self.scan])
            arr_norm = None if self.norm is None else np.atleast_1d(data_acc[self.norm])
            keys, sums, counts = worker_sparse_mean_sort(arr, arr_scan, self.decimals, arr_norm)
            out_dict[self.name] = (keys, sums, counts)


class Scan2VarWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        self.var = p['var']
        scans = p['scan']
        self.scan1, self.scan2 = scans[0], scans[1]
        decs = p.get('decimals', p.get('decimal'))
        self.dec1, self.dec2 = decs[0], decs[1]
        self.norm = p.get('norm')

    def accumulate(self, data_acc, out_dict):
        if self.var in data_acc and self.scan1 in data_acc and self.scan2 in data_acc:
            arr = np.atleast_1d(data_acc[self.var])
            s1 = np.atleast_1d(data_acc[self.scan1])
            s2 = np.atleast_1d(data_acc[self.scan2])
            arr_norm = None if self.norm is None else np.atleast_1d(data_acc[self.norm])
            k1, k2, sums_mat, counts_mat = worker_sparse_mean_sort2d(arr, s1, s2, self.dec1, self.dec2, arr_norm)
            out_dict[self.name] = (k1, k2, sums_mat, counts_mat)


class ScanHist1DWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        # arange: dict var->[start, stop, step]
        var = next(iter(p['arange']))
        self.var = var
        self.edges = np.arange(*p['arange'][var])
        self.scan = p['scan']
        dec = p.get('decimals', p.get('decimal'))
        self.decimals = dec
        self.norm = p.get('norm')

    def accumulate(self, data_acc, out_dict):
        if self.var in data_acc and self.scan in data_acc:
            arr = np.atleast_1d(data_acc[self.var])
            arr_scan = np.atleast_1d(data_acc[self.scan])
            arr_norm = None if self.norm is None else np.atleast_1d(data_acc[self.norm])
            H_sp, keys, num_arr = worker_sparse_sort1d_fast(arr, arr_scan, self.edges, self.decimals, arr_norm)
            out_dict[self.name] = (H_sp, keys, num_arr)


class Hist1DWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        var = next(iter(p['arange']))
        self.var = var
        self.edges = np.arange(*p['arange'][var])

    def accumulate(self, data_acc, out_dict):
        if self.var in data_acc:
            arr = np.atleast_1d(data_acc[self.var])
            H_sp, _ = worker_sparse_hist1d_fast(arr, self.edges)
            out_dict[self.name] = H_sp


class Hist2DWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        keys = list(p['arange'].keys())
        self.kx, self.ky = keys[0], keys[1]
        self.xedges = np.arange(*p['arange'][self.kx])
        self.yedges = np.arange(*p['arange'][self.ky])

    def accumulate(self, data_acc, out_dict):
        if self.kx in data_acc and self.ky in data_acc:
            x = np.atleast_1d(data_acc[self.kx])
            y = np.atleast_1d(data_acc[self.ky])
            H_sp, _, _ = worker_sparse_hist2d_fast(x, y, self.xedges, self.yedges)
            out_dict[self.name] = H_sp


class SingleLineWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        self.var = p['var']

    def accumulate(self, data_acc, out_dict):
        if self.var in data_acc:
            arr = np.asarray(data_acc[self.var])
            out_dict[self.name] = arr


class SingleImageWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        self.var = p['var']

    def accumulate(self, data_acc, out_dict):
        if self.var in data_acc:
            out_dict[self.name] = np.asarray(data_acc[self.var])


class RollAvg1DWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        self.var = p['var']

    def accumulate(self, data_acc, out_dict):
        if self.var in data_acc:
            arr = np.asarray(data_acc[self.var])
            out_dict[self.name]= arr


class SigBkg1DWorker(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        # names for signal and background plots
        self.name_sig = f"{name}_sig"
        self.name_bkg = f"{name}_bkg"
        plot_type = p['plot_type']

        # signal and background function dicts
        fsig = p['func_sig']
        fbkg = p['func_bkg']

        if plot_type == 'singleline_func':
            # single-line transform for signal, rolling avg for background
            self.worker_sig = SingleLineFuncWorkerPlot(self.name_sig, {'func': fsig})
            self.worker_bkg = RollAvg1DFuncWorkerPlot(self.name_bkg, {'func': fbkg})

        elif plot_type == 'rollavg1d_func':
            # rolling-average transform for both
            self.worker_sig = RollAvg1DFuncWorkerPlot(self.name_sig, {'func': fsig})
            self.worker_bkg = RollAvg1DFuncWorkerPlot(self.name_bkg, {'func': fbkg})

        elif plot_type == 'scan_var_func':
            # 1D scan-mean for signal and background
            psig = {
                'func': fsig,
                'func_scan': p['func_scan_sig'],
                'decimals': p.get('decimals', p.get('decimal'))
            }
            if 'func_norm_sig' in p:
                psig['func_norm'] = p['func_norm_sig']
            pbkg = {
                'func': fbkg,
                'func_scan': p['func_scan_bkg'],
                'decimals': p.get('decimals', p.get('decimal'))
            }
            if 'func_norm_bkg' in p:
                pbkg['func_norm'] = p['func_norm_bkg']
            self.worker_sig = ScanVarFuncWorkerPlot(self.name_sig, psig)
            self.worker_bkg = ScanVarFuncWorkerPlot(self.name_bkg, pbkg)

        elif plot_type == 'hist1d_func':
            # histogram transform for signal and background
            arange = p['arange_var']
            nsig = {'func': fsig, 'arange': arange}
            if 'func_norm_sig' in p:
                nsig['func_norm'] = p['func_norm_sig']
                nsig['norm_type'] = p.get('norm_type')
            nbkg = {'func': fbkg, 'arange': arange}
            if 'func_norm_bkg' in p:
                nbkg['func_norm'] = p['func_norm_bkg']
                nbkg['norm_type'] = p.get('norm_type')
            self.worker_sig = Hist1DFuncWorkerPlot(self.name_sig, nsig)
            self.worker_bkg = Hist1DFuncWorkerPlot(self.name_bkg, nbkg)

        else:
            raise ValueError(f"Unknown plot_type '{plot_type}'")

    def accumulate(self, data_acc, out_dict):
        self.worker_sig.accumulate(data_acc, out_dict)
        self.worker_bkg.accumulate(data_acc, out_dict)



class RollAvg1DFuncWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        fd = p['func']
        self.func = mk_func(fd.get('name'))
        self.func_args1 = fd.get('args1', [])
        self.func_args2 = fd.get('args2', [])

    def accumulate(self, data_acc, out_dict):
        
        args = [np.atleast_1d(data_acc[k]) for k in self.func_args1 if k in data_acc]
        if len(args) != len(self.func_args1): 
            return
        arr = self.func(*args, *self.func_args2)
        if arr is None: 
            return
        out_dict[self.name] = arr



class SingleLineFuncWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        fd = p['func']
        self.func = mk_func(fd.get('name'))
        self.func_args1 = fd.get('args1', [])
        self.func_args2 = fd.get('args2', [])

    def accumulate(self, data_acc, out_dict):
        args = [np.atleast_1d(data_acc[k]) for k in self.func_args1 if k in data_acc]
        if len(args) != len(self.func_args1): return
        arr = self.func(*args, *self.func_args2)
        if arr is None: 
            return
        out_dict[self.name] = np.atleast_1d(arr)


class Hist1DFuncWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        fd = p['func']
        self.func = mk_func(fd.get('name'))
        self.func_args1 = fd.get('args1', [])
        self.func_args2 = fd.get('args2', [])
        ar = p.get('arange', p.get('arange_var'))
        self.edges = np.arange(*ar)
        self.norm_type = p.get('norm_type')
        if self.norm_type:
            fn = p.get('func_norm')
            if fn:
                self.func_norm = mk_func(fn.get('name'))
                self.func_args1_norm = fn.get('args1', [])
                self.func_args2_norm = fn.get('args2', [])

    def accumulate(self, data_acc, out_dict):
        args = [np.atleast_1d(data_acc[k]) for k in self.func_args1 if k in data_acc]
        if len(args) != len(self.func_args1): return
        arr = self.func(*args, *self.func_args2)
        if arr is None: return
        H_sp, _ = worker_sparse_hist1d_fast(np.atleast_1d(arr), self.edges)
        out_dict[f'h1_{self.name}'] = H_sp
        if self.norm_type:
            args_n = [np.atleast_1d(data_acc[k]) for k in self.func_args1_norm if k in data_acc]
            if len(args_n) != len(self.func_args1_norm): return
            norm_arr = self.func_norm(*args_n, *self.func_args2_norm)
            if norm_arr is None: return
            val = np.sum(norm_arr) if self.norm_type=='sum' else norm_arr.size
            out_dict[f'norm_{self.name}'] = float(val)


class RollAvgFuncWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        fd = p['func']
        self.func = mk_func(fd.get('name'))
        self.func_args1 = fd.get('args1', [])
        self.func_args2 = fd.get('args2', [])

    def accumulate(self, data_acc, out_dict):
        args = [np.atleast_1d(data_acc[k]) for k in self.func_args1 if k in data_acc]
        if len(args) != len(self.func_args1): return
        arr = self.func(*args, *self.func_args2)
        if arr is None: return
        out_dict[self.name] = np.mean(np.atleast_1d(arr))


class ScanVarFuncWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        v = p['func']
        self.func_var = mk_func(v.get('name'))
        self.func_args1_var = v.get('args1', [])
        self.func_args2_var = v.get('args2', [])
        s = p['func_scan']
        self.func_scan = mk_func(s.get('name'))
        self.func_args1_scan = s.get('args1', [])
        self.func_args2_scan = s.get('args2', [])
        dec = p.get('decimals', p.get('decimal'))
        self.decimals = dec
        n = p.get('func_norm')
        if n:
            self.func_norm = mk_func(n.get('name'))
            self.func_args1_norm = n.get('args1', [])
            self.func_args2_norm = n.get('args2', [])
        else:
            self.func_norm = None

    def accumulate(self, data_acc, out_dict):
        args_v = [np.atleast_1d(data_acc[k]) for k in self.func_args1_var if k in data_acc]
        if len(args_v) != len(self.func_args1_var): return
        arr = np.atleast_1d(self.func_var(*args_v, *self.func_args2_var))
        args_s = [np.atleast_1d(data_acc[k]) for k in self.func_args1_scan if k in data_acc]
        if len(args_s) != len(self.func_args1_scan): return
        arr_scan = np.atleast_1d(self.func_scan(*args_s, *self.func_args2_scan))
        arr_norm = None
        if self.func_norm:
            args_n = [np.atleast_1d(data_acc[k]) for k in self.func_args1_norm if k in data_acc]
            if len(args_n) != len(self.func_args1_norm): return
            arr_norm = np.atleast_1d(self.func_norm(*args_n, *self.func_args2_norm))
        keys, sums, counts = worker_sparse_mean_sort(arr, arr_scan, self.decimals, arr_norm)
        out_dict[self.name] = (keys, sums, counts)


class Scan2VarFuncWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        # Main variable transform
        fv = p['func']
        self.func_var = mk_func(fv.get('name'))
        self.func_args1_var = fv.get('args1', [])
        self.func_args2_var = fv.get('args2', [])
        # First scan dimension
        fs1 = p['func_scan1']
        self.func_s1 = mk_func(fs1.get('name'))
        self.func_args1_s1 = fs1.get('args1', [])
        self.func_args2_s1 = fs1.get('args2', [])
        # Second scan dimension
        fs2 = p['func_scan2']
        self.func_s2 = mk_func(fs2.get('name'))
        self.func_args1_s2 = fs2.get('args1', [])
        self.func_args2_s2 = fs2.get('args2', [])
        # Binning precision for each scan
        decimals = p.get('decimal', p.get('decimals', [5, 5]))
        self.dec1, self.dec2 = decimals[0], decimals[1]
        # Optional normalization
        fn = p.get('func_norm')
        if fn:
            self.func_norm = mk_func(fn.get('name'))
            self.func_args1_norm = fn.get('args1', [])
            self.func_args2_norm = fn.get('args2', [])
        else:
            self.func_norm = None

    def accumulate(self, data_acc: dict, out_dict: dict):# -> None:
        # Compute main variable array
        args_v = []
        for k in self.func_args1_var:
            if k not in data_acc:
                return
            args_v.append(np.atleast_1d(data_acc[k]))
        arr = np.atleast_1d(self.func_var(*args_v, *self.func_args2_var))
        # Compute first scan axis
        args1 = []
        for k in self.func_args1_s1:
            if k not in data_acc:
                return
            args1.append(np.atleast_1d(data_acc[k]))
        s1 = np.atleast_1d(self.func_s1(*args1, *self.func_args2_s1))
        # Compute second scan axis
        args2 = []
        for k in self.func_args1_s2:
            if k not in data_acc:
                return
            args2.append(np.atleast_1d(data_acc[k]))
        s2 = np.atleast_1d(self.func_s2(*args2, *self.func_args2_s2))
        # Compute normalization array if provided
        arr_norm = None
        if self.func_norm:
            args_n = []
            for k in self.func_args1_norm:
                if k not in data_acc:
                    return
                args_n.append(np.atleast_1d(data_acc[k]))
            arr_norm = np.atleast_1d(self.func_norm(*args_n, *self.func_args2_norm))
        # Compute sparse 2D mean
        k1, k2, sums_mat, counts_mat = worker_sparse_mean_sort2d(
            arr, s1, s2, self.dec1, self.dec2, arr_norm
        )
        out_dict[self.name] = (k1, k2, sums_mat, counts_mat)


class ScanHist1DFuncWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        # function transforming the main variable
        fv = p['func']
        self.func_var = mk_func(fv.get('name'))
        self.func_args1_var = fv.get('args1', [])
        self.func_args2_var = fv.get('args2', [])
        # function computing scan variable
        fs = p['func_scan']
        self.func_scan = mk_func(fs.get('name'))
        self.func_args1_scan = fs.get('args1', [])
        self.func_args2_scan = fs.get('args2', [])
        # bin edges definition
        ar = p['arange_var']
        self.edges = np.arange(*ar)
        # decimals for grouping
        self.decimals = p.get('decimals', p.get('decimal', 5))
        # optional normalization
        fn = p.get('func_norm')
        if fn:
            self.func_norm = mk_func(fn.get('name'))
            self.func_args1_norm = fn.get('args1', [])
            self.func_args2_norm = fn.get('args2', [])
        else:
            self.func_norm = None

    def accumulate(self, data_acc: dict, out_dict: dict):# -> None:
        # gather main var
        args_v = []
        for k in self.func_args1_var:
            if k not in data_acc:
                return
            args_v.append(np.atleast_1d(data_acc[k]))
        arr = np.atleast_1d(self.func_var(*args_v, *self.func_args2_var))
        # gather scan var
        args_s = []
        for k in self.func_args1_scan:
            if k not in data_acc:
                return
            args_s.append(np.atleast_1d(data_acc[k]))
        arr_scan = np.atleast_1d(self.func_scan(*args_s, *self.func_args2_scan))
        # gather norm var if any
        arr_norm = None
        if self.func_norm:
            args_n = []
            for k in self.func_args1_norm:
                if k not in data_acc:
                    return
                args_n.append(np.atleast_1d(data_acc[k]))
            arr_norm = np.atleast_1d(self.func_norm(*args_n, *self.func_args2_norm))
        # compute sparse histogram
        H_sp, keys, counts = worker_sparse_sort1d_fast(
            arr, arr_scan, self.edges, self.decimals, arr_norm
        )
        out_dict[self.name] = (H_sp, keys, counts)


class Hist2DFuncWorkerPlot(BaseWorkerPlot):
    def __init__(self, name, p):
        super().__init__(name)
        # function for x-axis
        fx = p['func_x']
        self.func_x = mk_func(fx.get('name'))
        self.func_args1_x = fx.get('args1', [])
        self.func_args2_x = fx.get('args2', [])
        # function for y-axis
        fy = p['func_y']
        self.func_y = mk_func(fy.get('name'))
        self.func_args1_y = fy.get('args1', [])
        self.func_args2_y = fy.get('args2', [])
        # bin edges for x and y
        ar = p['arange_var']
        # expecting keys: two entries in the dict
        keys = list(ar.keys())
        xkey, ykey = keys[0], keys[1]
        self.xedges = np.arange(*ar[xkey])
        self.yedges = np.arange(*ar[ykey])

    def accumulate(self, data_acc: dict, out_dict: dict):# -> None:
        # compute x values
        args_x = []
        for k in self.func_args1_x:
            if k not in data_acc:
                return
            args_x.append(np.atleast_1d(data_acc[k]))
        x = np.atleast_1d(self.func_x(*args_x, *self.func_args2_x))
        # compute y values
        args_y = []
        for k in self.func_args1_y:
            if k not in data_acc:
                return
            args_y.append(np.atleast_1d(data_acc[k]))
        y = np.atleast_1d(self.func_y(*args_y, *self.func_args2_y))
        # compute 2D sparse histogram
        H_sp2d, _, _ = worker_sparse_hist2d_fast(x, y, self.xedges, self.yedges)
        out_dict[self.name] = H_sp2d
