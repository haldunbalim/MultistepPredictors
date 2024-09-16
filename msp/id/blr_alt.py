import numpy as np
from d2pc.id_em import LTISystem
from d2pc.utils import block_diag
from msp.utils import dense2bandedh, dense2diags, compute_msp


def compute_sigma_banded_alt(sys: LTISystem, K: int, Ls: np.ndarray, calS: np.ndarray):
    """
        Compute the banded form of the covariance matrix Sigma
        - sys: LTISystem system
        - K: int multi-step horizon
        - Ls:  np.ndarray KF gains (T, nx, nx) 
        - calS: np.ndarray (T*ny, T*ny) block diagonal array of innovation covariance matrices
    """
    # base case
    if K == 1:
        return dense2bandedh(calS, sys.ny, lower=True)
    Gw = compute_msp(sys, K)[1]
    calGw = compute_calGw_alt(Gw, K, Ls)
    return dense2bandedh(calGw @ calS @ calGw.T, (K+1) * sys.ny, lower=True)


def compute_calGw_alt(Gw: np.ndarray, K: int, Ls: np.ndarray):
    """
        Compute the banded diagonal matrix calGw
        - Gw: np.ndarray (T*ny, T*ny)
        - K: int multi-step horizon
        - Ls: np.ndarray KF gains (T, nx, nx)
    """
    T, ny = Ls.shape[0], Ls.shape[-1]
    nx = Gw.shape[1] // K
    calGw = np.zeros(((T-K) * ny, T * ny))
    conv = False
    for i in range(T-K):
        if conv:
            calGw[i*ny:(i+1)*ny, i*ny: (i+K+1) * ny] = Gw_curr
        else:
            Gw_curr = Gw @ block_diag(*Ls[i:i+K])
            Gw_curr = np.hstack([Gw_curr, np.eye(ny)])
            calGw[i*ny:(i+1)*ny, i*ny: (i+K+1) * ny] = Gw_curr
            conv = i > 0 and np.allclose(Ls[i-1], Ls[i], atol=1e-12, rtol=0)
    return dense2diags(calGw, ny-1, K*ny)
