from .blr_kf import *
from msp.utils import *
from d2pc import lchol

# returns Gw_w Q Gw_w^\top + Gw_v R Gw_v^\top
def compute_sigma_banded_jk(sys: MSPSystem, K:int, T:int):
    if isinstance(sys, CanonSystem):
        sys = convert_canon_to_IO(sys)
    elif isinstance(sys, ObsCanonSystem):
        sys = convert_obs_canon_to_IO(sys)
    elif not isinstance(sys, IOSystem):
        raise Exception("Only MSPSystemAllowed")
    
    ny, nu = sys.ny, sys.nu
    order = (sys.nx + sys.nu) // (sys.ny + sys.nu)
    G0u, Gw = compute_msp(sys, K, use_E=True)
    G0 = G0u[:, :order*ny+order*nu-nu]

    Q_lchol, R_lchol = lchol(sys.Q), lchol(sys.R)
    calGw = compute_calGw_jk(Gw, order, T, K, Q_lchol=Q_lchol)
    calGv = compute_calGv_jk(G0, order, T, K, R_lchol=R_lchol)
    Sigma_bd = calGw @ calGw.T + calGv @ calGv.T
    return dense2bandedh(Sigma_bd, (K + order) * ny, lower=True)

def compute_calGw_jk(Gw, order, T, K,  Q_lchol=None):
    ny = Gw.shape[0]
    calGw = np.zeros(((T - K + 1 - order), ny, (T - order) * ny))
    if Q_lchol is not None:
        Gw = Gw @ block_diag(Q_lchol, K)
    for i in range(T - K + 1 - order):
        calGw[i, :, i*ny:i*ny+K*ny] = Gw
    calGw = calGw.reshape((-1, (T-order)*ny))
    return dense2diags(calGw, ny-1, (K-1)*ny)

def compute_calGv_jk(G0, order, T, K, R_lchol=None):
    ny = G0.shape[0]
    nu = (G0.shape[1] - ny*order) // (order-1)
    G0 = G0[:, order*nu-nu:]
    calGv = np.zeros(((T - K + 1 - order), ny, T * ny))
    if R_lchol is None:
        R_lchol = np.eye(ny)
    else:
        G0 = G0 @ block_diag(R_lchol, order)
    for i in range(T - K + 1 - order):
        calGv[i, :, i*ny:i*ny+order*ny] = -G0
        calGv[i, :, (i+K+order-1)*ny: (i+K+order)*ny] = R_lchol
    calGv = calGv.reshape((-1, T * ny))
    return dense2diags(calGv, ny-1, (K+order-1)*ny)
