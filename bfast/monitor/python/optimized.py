
import numpy as np
from numba import jit, prange

@jit(nopython=True)
def _log_plus_scalar(val):
    if val > np.e:
        return np.log(val)
    return 1.0

@jit(nopython=True)
def fit_single_optimized(y, X, n, hfrac, k, lam, mapped_indices):
    """
    Optimized version of fit_single using Numba.
    
    y: 1D array (N,)
    X: 2D array (n_features, N)
    n: int (start of monitoring period index in full array)
    hfrac: float
    k: int
    lam: float
    mapped_indices: 1D array (N,)
    """
    N = y.shape[0]
    
    # 1. Handle NaNs
    nans = np.isnan(y)
    # cumsum of nans
    # Manual cumsum to be safe across numba versions or just use np.cumsum
    num_nans = np.cumsum(nans) # Numba supports this
    
    # compute new limits
    # ns = self.n - num_nans[self.n - 1]
    # Check bounds
    idx_n_minus_1 = n - 1
    if idx_n_minus_1 < 0: idx_n_minus_1 = 0
    
    ns = n - num_nans[idx_n_minus_1]
    h = int(float(ns) * hfrac)
    Ns = N - num_nans[N - 1]
    
    if ns <= 5 or Ns - ns <= 5:
        return -2, 0.0, 0.0, Ns

    # 2. Filter data
    # Create valid mask
    valid_mask = ~nans
    
    # Filter y
    y_nn = y[valid_mask]
    
    # Filter X
    # X is (features, N)
    # We want X_nn to be (features, Ns)
    # Numba doesn't support fancy indexing on 2D arrays easily in all cases,
    # but boolean indexing on columns should work: X[:, mask]
    # Let's try explicit construction if that fails, but X[:, mask] is standard.
    # To be safe for nopython mode, manual copy might be needed if X[:, mask] fails compilation.
    # But let's assume it works. If not, I'll fix it.
    
    # Actually, X[:, mask] creates a copy.
    X_nn = X[:, valid_mask]
    
    # Split into history and monitoring
    # History: first ns points
    X_nn_h = X_nn[:, :ns]
    y_nn_h = y_nn[:ns]
    
    # 3. Linear Regression
    # We want to solve for beta: X_nn_h.T @ beta = y_nn_h -> No, y = X @ beta
    # X_nn_h is (features, samples). We want samples as rows for standard notation.
    # Sklearn: fit(X, y) where X is (samples, features).
    # base.py: fit(X_nn_h.T, y_nn_h).
    # So we solve: (X_nn_h.T) @ beta ~ y_nn_h
    # Normal eq: beta = inv( (X_nn_h.T).T @ (X_nn_h.T) ) @ (X_nn_h.T).T @ y_nn_h
    #           beta = inv( X_nn_h @ X_nn_h.T ) @ X_nn_h @ y_nn_h
    
    Xt = X_nn_h.T # (samples, features)
    # XtX = X_nn_h @ Xt # (features, features)
    XtX = np.dot(X_nn_h, Xt)
    # Xty = X_nn_h @ y_nn_h # (features,)
    Xty = np.dot(X_nn_h, y_nn_h)
    
    # Solve
    # beta = np.linalg.solve(XtX, Xty)
    # Use lstsq for stability (equivalent to fit_intercept=False which uses internal solver)
    beta = np.linalg.lstsq(XtX, Xty)[0]
    
    # 4. Predict on ALL valid data
    # y_pred = beta @ X_nn (features vector @ features x samples matrix) -> (samples,)
    # Dimensions: (features,) @ (features, samples) -> (samples,)
    y_pred = np.dot(beta, X_nn)
    
    y_error = y_nn - y_pred
    
    # 5. MOSUM Process
    # err_cs = np.cumsum(y_error[ns - h:Ns + 1])
    # Slice indices
    start_slice = ns - h
    if start_slice < 0: start_slice = 0
    # end_slice is Ns + 1 (implicit end of array or just Ns)
    
    # Extract error segment
    # Note: y_error has length Ns.
    # In base.py: y_error[ns - h : Ns + 1]
    # Python slicing clamps end.
    
    y_error_slice = y_error[start_slice:]
    err_cs = np.cumsum(y_error_slice)
    
    # mosum_nn = err_cs[h:] - err_cs[:-h]
    # Length of err_cs is len(y_error_slice).
    # We need h < len(err_cs).
    
    if h >= len(err_cs):
        mosum_nn = np.zeros(1, dtype=np.float32) # Should not happen usually
    else:
        mosum_nn = err_cs[h:] - err_cs[:-h]
    
    # Sigma
    # sigma = np.sqrt(np.sum(y_error[:ns] ** 2) / (ns - (2 + 2 * self.k)))
    # Assuming trend=True (2 parameters: intercept + slope? base.py says "Intercept", "trend").
    # If trend=False, base.py selects indices.
    # But line 259 uses fixed `(ns - (2 + 2 * self.k))` regardless of trend?
    # Wait, checking base.py line 259...
    # Yes, it uses `2 + 2 * self.k`. This implies 2 trend params + 2*k harmonic params.
    # Even if trend=False?
    # base.py line 306: if trend: X has 2 rows (ones, indices). Else: X has 1 row (ones).
    # Then adds 2*k rows.
    # So parameters = (1 or 2) + 2*k.
    # The divisor should be ns - num_params.
    # If trend is False, num_params = 1 + 2*k.
    # If trend is True, num_params = 2 + 2*k.
    # base.py seems to hardcode 2+2*k. I should follow base.py to reproduce exact results.
    
    sigma2 = np.sum(y_error[:ns]**2) / (ns - (2 + 2 * k))
    sigma = np.sqrt(sigma2)
    
    if sigma == 0:
        # Avoid division by zero
        factor = 0.0
    else:
        factor = 1.0 / (sigma * np.sqrt(ns))
        
    mosum_nn_scaled = factor * mosum_nn
    
    # 6. Compute stats
    mean = np.mean(mosum_nn_scaled)
    
    # Magnitude
    # magnitude = np.median(y_error[ns:])
    # y_error[ns:] is the error in monitoring period
    if len(y_error) > ns:
        magnitude = np.median(y_error[ns:])
    else:
        magnitude = 0.0
    
    # 7. Check Breaks
    # We need to map mosum_nn back to the monitoring timeline to check against bounds.
    # OR map bounds to mosum_nn.
    
    # Bounds depend on `mapped_indices`.
    # a = mapped_indices[n:] / mapped_indices[n-1]
    
    # We need the values of mapped_indices corresponding to the mosum_nn points?
    # In base.py: 
    # mosum = np.repeat(np.nan, N - self.n)
    # mosum[val_inds[:Ns - ns]] = mosum_nn
    # breaks = np.abs(mosum) > bounds
    
    # `bounds` is calculated for ALL points in monitoring period (length N-n).
    # `mosum` has values only at `val_inds`.
    # So we compare `mosum_nn[i]` with `bounds[val_inds[i]]`.
    
    # Reconstruct val_inds (indices in monitoring period 0..N-n-1)
    # val_inds in base.py logic:
    # 1. Get valid indices of original array: valid_inds_orig
    # 2. Keep those >= ns (relative to start of history? No, relative to start of history in VALID array)
    # base.py: val_inds = np.array(range(N))[~nans]
    #          val_inds = val_inds[ns:]
    #          val_inds -= self.n
    
    # My `ns` is count of valid history points.
    # `val_inds` are indices in the original array that are valid.
    # We want indices that are in the monitoring period (index >= n).
    
    valid_indices_orig = np.arange(N)[valid_mask]
    
    # We need to find which valid indices are in the monitoring period.
    # The monitoring period starts at `n` in the original array.
    
    # Filter valid_indices_orig >= n
    valid_indices_mon = valid_indices_orig[valid_indices_orig >= n]
    
    # We also need to know which elements of `mosum_nn` correspond to these.
    # `mosum_nn` is derived from `y_error` which contains all valid points.
    # `y_error` first `ns` points are history.
    # So `y_error[ns:]` corresponds to valid points in monitoring period?
    # Yes, because `ns` is calculated as `n - num_nans_before_n`.
    # So the first `ns` valid points come from the first `n` original points.
    # The remaining valid points come from `n..N`.
    
    # So `mosum_nn` corresponds 1-to-1 with `valid_indices_mon`.
    # And `val_inds` (relative to n) would be `valid_indices_mon - n`.
    
    # Calculate Bounds
    # a = mapped_indices[n:] / float(mapped_indices[n - 1])
    denom = float(mapped_indices[n - 1])
    if denom == 0: denom = 1.0
    
    a_full = mapped_indices[n:] / denom
    
    # bounds_full = lam * sqrt(log_plus(a_full))
    # We can compute bounds just for the valid indices to save time, or precompute.
    # Since we are inside the loop and mapped_indices is constant, maybe precompute outside?
    # But for now, let's compute inside or just the needed ones.
    
    # We need to compare `mosum_nn[i]` with `bounds_full[valid_indices_mon[i] - n]`.
    
    first_break = -1
    
    # Iterate over valid monitoring points
    # Limit by length of mosum_nn (which should match valid_indices_mon length)
    
    limit = min(len(mosum_nn_scaled), len(valid_indices_mon))
    
    for i in range(limit):
        mon_idx = valid_indices_mon[i] - n
        
        # Compute bound for this specific index
        val_a = a_full[mon_idx]
        log_val = 1.0
        if val_a > np.e:
            log_val = np.log(val_a)
        
        bound = lam * np.sqrt(log_val)
        
        if np.abs(mosum_nn_scaled[i]) > bound:
            first_break = mon_idx # Return index relative to monitoring start
            break
            
    return first_break, mean, magnitude, Ns

@jit(nopython=True, parallel=True)
def fit_optimized_loop(data_flat, X, n, hfrac, level, k, trend, lam, mapped_indices, 
                       breaks_out, means_out, magnitudes_out, valids_out):
    """
    Parallel loop over pixels.
    data_flat: (N_pixels, N_time)
    """
    n_pixels = data_flat.shape[0]
    
    for i in prange(n_pixels):
        y = data_flat[i, :]
        brk, mean, mag, val = fit_single_optimized(y, X, n, hfrac, k, lam, mapped_indices)
        
        breaks_out[i] = brk
        means_out[i] = mean
        magnitudes_out[i] = mag
        valids_out[i] = val

