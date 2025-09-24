import numpy as np
from scipy import sparse
from typing import Dict, Tuple, Sequence, Optional

# Worker function for two scanning variables
def worker_sparse_mean_sort2d(
    data: np.ndarray,
    scan1: np.ndarray,
    scan2: np.ndarray,
    decimals1: int = 5,
    decimals2: int = 5,
    arr_norm: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group `data` by rounded (scan1, scan2) pairs, compute sum and weighted counts.

    Returns:
      sorted_keys1 : np.ndarray (G1,)
      sorted_keys2 : np.ndarray (G2,)
      sums_mat     : np.ndarray (G1, G2)
      counts_mat   : np.ndarray (G1, G2)
    """
    if not (data.shape == scan1.shape == scan2.shape):
        raise ValueError("`data`, `scan1`, `scan2` must have the same shape")
    if arr_norm is not None and arr_norm.shape != data.shape:
        raise ValueError("`arr_norm` must have the same shape as inputs")

    # Round and invert indices
    s1r = np.round(scan1, decimals1)
    s2r = np.round(scan2, decimals2)
    sorted1, inv1 = np.unique(s1r, return_inverse=True)
    sorted2, inv2 = np.unique(s2r, return_inverse=True)
    G1, G2 = sorted1.size, sorted2.size

    # Flatten 2D indices
    flat = inv1 * G2 + inv2

    # Compute sums
    sums_flat = np.bincount(flat, weights=data, minlength=G1*G2)
    sums_mat = sums_flat.reshape(G1, G2)

    # Compute counts (weighted)
    if arr_norm is None:
        counts_flat = np.bincount(flat, minlength=G1*G2)
    else:
        counts_flat = np.bincount(flat, weights=arr_norm.astype(float), minlength=G1*G2)
    counts_mat = counts_flat.reshape(G1, G2)

    return sorted1, sorted2, sums_mat, counts_mat
    
# Worker-side function: group a single variable and compute sums and counts per scan-key
def worker_sparse_mean_sort(
    data: np.ndarray,
    scan: np.ndarray,
    decimals: int = 5,
    arr_norm: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data.shape != scan.shape:
        raise ValueError("`data` and `scan` must have the same shape")
    if arr_norm is not None and arr_norm.shape != data.shape:
        raise ValueError("`arr_norm` must match input shapes")

    # round and invert
    scan_r = np.round(scan, decimals)
    sorted_keys, inv = np.unique(scan_r, return_inverse=True)
    G = sorted_keys.size

    # compute sums and counts
    sums   = np.bincount(inv, weights=data, minlength=G)
    if arr_norm is None:
        counts = np.bincount(inv, minlength=G).astype(float)
    else:
        counts = np.bincount(inv, weights=arr_norm.astype(float), minlength=G)

    return sorted_keys, sums, counts
    
def worker_sparse_sort1d_fast(
    data: np.ndarray,
    scan: np.ndarray,
    data_edges: np.ndarray,
    decimals: int = 5,
    arr_norm: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group `data` by rounded scan-values and build sparse 1D histograms.
    Returns H_sp (G×M), sorted_keys (G,), num_arr (G,).
    """
    if data.shape != scan.shape:
        raise ValueError("`data` and `scan` must have the same shape")
    if arr_norm is not None and arr_norm.shape != data.shape:
        raise ValueError("`arr_norm` must match input shapes")

    # 1) group indices
    scan_r = np.round(scan, decimals)
    sorted_keys, inv = np.unique(scan_r, return_inverse=True)
    G = sorted_keys.size
    M = data_edges.size - 1

    # 2) bin data
    dx = data_edges[1] - data_edges[0]
    bins = ((data - data_edges[0]) / dx).astype(int)
    mask = (bins >= 0) & (bins < M)
    inv_m, bins = inv[mask], bins[mask]

    # 3) flatten for unique
    flat = inv_m * M + bins
    pos, cnts = np.unique(flat, return_counts=True)
    rows = pos // M
    cols = pos % M

    H_sp = sparse.coo_matrix((cnts, (rows, cols)), shape=(G, M))

    # 4) counts per group
    if arr_norm is None:
        num_arr = np.bincount(inv_m, minlength=G)
    else:
        num_arr = np.bincount(inv_m, weights=arr_norm[mask].astype(float), minlength=G)

    return H_sp, sorted_keys, num_arr


# -------------------------------------------------------------------
# STEP 1: single‐time bin‐edge creation using np.arange
# -------------------------------------------------------------------

def init_hist1d(arange: Sequence[float]) -> np.ndarray:
    """
    arange = [start, stop, step]
    returns edges = [start, start+step, …, stop]
    """
    start, stop, step = arange
    # include the final edge so that the last bin is [stop–step, stop]
    return np.arange(start, stop + step, step)

def init_hist2d(x_arange: Sequence[float], y_arange: Sequence[float]
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_arange, y_arange = [start, stop, step] for each axis.
    """
    x0, x1, dx = x_arange
    y0, y1, dy = y_arange
    xedges = np.arange(x0, x1 + dx, dx)
    yedges = np.arange(y0, y1 + dy, dy)
    return xedges, yedges

# -------------------------------------------------------------------
# STEP 2: fast 1D sparse‐hist worker that reuses precomputed edges
# -------------------------------------------------------------------

def worker_sparse_hist1d_fast(
    data: np.ndarray,
    edges: np.ndarray
) -> Tuple[sparse.coo_matrix, np.ndarray]:
    """
    data : 1D array of samples
    edges: bin‐edge array (length N+1)
    
    Returns
    -------
    H_sp  : sparse.coo_matrix of shape (N,1)
    edges : as passed in
    """
    # number of bins and bin width
    N = edges.size - 1
    dx = edges[1] - edges[0]

    # arithmetic binning
    idx = ((data - edges[0]) / dx).astype(int)
    mask = (idx >= 0) & (idx < N)
    idx = idx[mask]

    # C‐level unique/counts
    bins_used, counts = np.unique(idx, return_counts=True)

    # build sparse (N×1) histogram
    H_sp = sparse.coo_matrix(
        (counts, (bins_used, np.zeros_like(bins_used))),
        shape=(N, 1)
    )
    return H_sp, edges


def group_sparse_hist1d_fast(
    data1: np.ndarray,
    data2: np.ndarray,
    edges: np.ndarray,
    decimals: int = 5,
    arr_norm: Optional[np.ndarray] = None
) -> Tuple[Dict[float, sparse.coo_matrix], Dict[float, float]]:
    """
    Groups data1 by rounded values of data2 and computes sparse 1D histograms for each group,
    also computing either the count or a weighted sum per group.

    Parameters
    ----------
    data1 : np.ndarray
        1D array of samples to histogram.
    data2 : np.ndarray
        1D array of same length as data1; values used to group data1.
    edges : np.ndarray
        Bin-edge array (length N+1) for the histogram.
    decimals : int, optional
        Number of decimal places to round data2 before grouping (default is 5).
    arr_norm : Optional[np.ndarray], optional
        1D array of same length as data1/data2. If provided, num_dict[val] is the sum
        of arr_norm at those indices; if None, num_dict[val] is simply the count of items.

    Returns
    -------
    H_sp_dict : Dict[float, sparse.coo_matrix]
        Maps each unique rounded data2 value to its sparse histogram of data1.
    num_dict : Dict[float, float]
        Maps each unique rounded data2 value to either the count or weighted sum.
    """
    if data1.shape != data2.shape:
        raise ValueError("data1 and data2 must have the same shape.")
    if arr_norm is not None and arr_norm.shape != data1.shape:
        raise ValueError("arr_norm must be the same shape as data1/data2.")

    # Round data2
    data2_rounded = np.round(data2, decimals)

    # Find unique rounded values
    unique_vals = np.unique(data2_rounded)

    # Prepare output dicts
    H_sp_dict: Dict[float, sparse.coo_matrix] = {}
    num_dict: Dict[float, float] = {}

    for val in unique_vals:
        inds = np.nonzero(data2_rounded == val)[0]
        if arr_norm is None:
            # simple count
            num_dict[val] = float(inds.size)
        else:
            # weighted sum
            num_dict[val] = float(arr_norm[inds].sum())

        # build sparse histogram for this slice of data1
        H_sp, _ = worker_sparse_hist1d_fast(data1[inds], edges)
        H_sp_dict[val] = H_sp

    return H_sp_dict, num_dict




# -------------------------------------------------------------------
# STEP 3: fast 2D sparse‐hist worker that reuses precomputed edges
# -------------------------------------------------------------------

def worker_sparse_hist2d_fast(
    x: np.ndarray,
    y: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray
) -> Tuple[sparse.coo_matrix, np.ndarray, np.ndarray]:
    """
    x, y     : 1D arrays of same length
    xedges, yedges : bin‐edge arrays for each axis
    
    Returns
    -------
    H_sp   : sparse.coo_matrix of shape (nx, ny)
    xedges : as passed in
    yedges : as passed in
    """
    nx = xedges.size - 1
    ny = yedges.size - 1
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    # arithmetic binning on each axis
    ix = ((x - xedges[0]) / dx).astype(int)
    iy = ((y - yedges[0]) / dy).astype(int)

    # mask out‐of‐range
    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[mask], iy[mask]

    # flatten 2D bins → 1D for unique/count
    flat = ix * ny + iy
    pos, counts = np.unique(flat, return_counts=True)

    # decode back to 2D indices
    i2 = pos // ny
    j2 = pos % ny

    H_sp = sparse.coo_matrix((counts, (i2, j2)), shape=(nx, ny))
    return H_sp, xedges, yedges

# -------------------------------------------------------------------
# STEP 4: gathering into dense arrays (no change!)
# -------------------------------------------------------------------

def gather_dense_hist1d_fast(
    dense_hist: np.ndarray,
    sparse_hist: sparse.coo_matrix
) -> None:
    np.add.at(dense_hist, sparse_hist.row, sparse_hist.data)

def gather_dense_hist2d_fast(
    dense_hist: np.ndarray,
    sparse_hist: sparse.coo_matrix
) -> None:
    np.add.at(dense_hist, (sparse_hist.row, sparse_hist.col), sparse_hist.data)

# -------------------------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------------------------

# Suppose from your YAML you pulled:
#   n[l].arange → nl = [0, 10, 1]
#   x-y[l].arange → xl = [-60, 60, 1], yl = [-60, 60, 1]

# # 1) In each worker, once:
# nl_edges = init_hist1d([0, 10, 1])
# xy_edges = init_hist2d([-60, 60, 1], [-60, 60, 1])

# # 2) For each chunk:
# H1_sp, _      = worker_sparse_hist1d_fast(data_chunk, nl_edges)
# Hxy_sp, _, _  = worker_sparse_hist2d_fast(x_chunk, y_chunk, *xy_edges)

# # 3) In the gatherer, once:
# dense_nl = np.zeros(nl_edges.size - 1,       dtype=int)
# dense_xy = np.zeros((xy_edges[0].size - 1,
#                      xy_edges[1].size - 1), dtype=int)

# # 4) As sparse pieces arrive:
# gather_dense_hist1d_fast(dense_nl, H1_sp)
# gather_dense_hist2d_fast(dense_xy, Hxy_sp)
