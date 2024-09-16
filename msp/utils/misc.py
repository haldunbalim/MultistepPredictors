import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from d2pc import LTISystem, psd_inverse
import scipy.linalg as la
from scipy.stats import norm
import control as ct
import cvxpy as cp
import pickle
import os
import glob
from datetime import datetime
import logging

def plot_uq(delta_mins:np.ndarray, ax=None, verbose=0):
    """
        Plot the uncertainty quantification from delta mins
    """
    ret_fig = False
    if ax is None:
        ret_fig = True
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 100)
    y = []
    for _x in x:
        y.append(np.mean(delta_mins <= _x))
    ax.plot(x, y)
    ax.grid()
    ax.plot([0, 1], [0, 1], c="black", linestyle="--")
    if verbose:
        for e in list(np.arange(.5, 1, .05)) + [.99]:
            print(
                f"Probability of true system being inside Theta_{e:.3f} is {np.mean(delta_mins <= e):.3f}")
    if ret_fig:
        return fig
    

def dense2bandedh(mat, band_sz: int, lower=False):
    """
        Convert a hermitian matrix to a banded matrix
        - mat: np.ndarray (n, n) dense matrix
        - band_sz: int size of the band (includes main diagonal)
        - lower: bool if True, the lower part of the matrix is used
    """
    assert mat.shape[0] == mat.shape[1]
    banded = np.zeros((band_sz, mat.shape[1]))
    for i in range(band_sz):
        if lower:
            banded[i, :banded.shape[1]-i] = mat.diagonal(-i)
        else:
            banded[banded.shape[0]-i-1, i:] = mat.diagonal(i)
    return banded

def bandedh2dense(banded, lower=False):
    """
        Convert a banded hermitian matrix to a dense matrix
        - banded: np.ndarray (n, m) banded matrix
        - lower: bool if True, the lower part of the matrix is used
    """
    if lower:
        ls = [b[:banded.shape[1]-i] for i, b in enumerate(banded)]
        idx = list(reversed(range(1-len(ls), 1)))
    else:
        ls = [b[i:] for i, b in enumerate(reversed(banded))]
        idx = range(len(ls))
    ls[0] = ls[0]/2
    mat = sp.diags(ls, idx)
    mat = mat + mat.T
    return mat.todense()

def dense2diags(banded, l_band_sz, u_band_sz):
    offsets = list(range(-l_band_sz, u_band_sz+1))
    ls = []
    for i in offsets:
        ls.append(banded.diagonal(i))
    return sp.diags(ls, offsets, shape=banded.shape)

def compute_msp(sys, K, use_E=False):
    assert K >= 1
    Apows = [np.eye(sys.nx)]
    for _ in range(K-1):
        Apows.append(sys.A @ Apows[-1])
    G0u = np.hstack([sys.A @ Apows[-1]] +
                    [Apow @ sys.B for Apow in reversed(Apows)])
    G0u_true = sys.C @ G0u
    if use_E:
        Gw_true = sys.C @ np.hstack([Apow @ sys.E for Apow in reversed(Apows)])
    else:
        Gw_true = sys.C @ np.hstack(list(reversed(Apows)))
    return G0u_true, Gw_true


def tf2_siso_arx(tf, dt, q=1e-4, r=1e-4, p0=1e-4):
    ss = ct.c2d(ct.tf2ss(tf), dt)
    canon, T = ct.canonical_form(ss, "observable")
    Tprime = la.inv(np.rot90(np.eye(3))) @ T
    Tprime_inv = la.inv(Tprime)
    A = Tprime @ ss.A @ Tprime_inv
    A[:, :-1] = np.round(A[:, :-1], 0)
    B = Tprime @ ss.B

    order = A.shape[0]
    Q, R, P0 = np.eye(1)*q, np.eye(1)*r, np.eye(order)*p0
    E = np.array([0] * (order-1) + [1])[:, None]
    C = E.T
    mu0 = np.zeros(order)
    return LTISystem(A, B, C, E, Q, R, mu0, P0)


def generate_random_siso_arx(order, q=1e-4, r=1e-4, p0=1e-4):
    ss = ct.drss(order, 1, 1)
    canon, T = ct.canonical_form(ss, "observable")
    Tprime = la.inv(np.rot90(np.eye(3))) @ T
    Tprime_inv = la.inv(Tprime)
    A = Tprime @ ss.A @ Tprime_inv
    A[:, :-1] = np.round(A[:, :-1], 0)
    B = Tprime @ ss.B

    Q, R, P0 = np.eye(1)*q, np.eye(1)*r, np.eye(order)*p0
    E = np.array([0] * (order-1) + [1])[:, None]
    C = E.T
    mu0 = np.zeros(order)
    return LTISystem(A, B, C, E, Q, R, mu0, P0)

def shifter(n, m):
    return np.hstack([np.zeros((n*m-n, n)), np.eye(n*m-n)])


def ss_to_arx_params(A, B, C):
    """
        Converts arbitrary state space model to observer canonical form
    """
    ny, nx = C.shape
    Tu = cp.Variable((nx-ny, nx))
    T = cp.vstack([Tu, C])
    Acs = cp.Variable((nx, ny))
    AcC = cp.vstack([Acs[i*ny:i*ny+ny, :]@C for i in range(nx // ny)])
    constraints = [T @ A - AcC == cp.vstack([np.zeros((ny, nx)), Tu])]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        raise Exception("Optimization failed")
    T = T.value
    Tinv = la.inv(T)
    A = T @ A @ Tinv
    B = T @ B
    order = nx // ny
    As, Bs = np.array(np.vsplit(A[:, -ny:], order)), np.array(np.vsplit(B, order))
    return As, Bs


# E.A. Yildirim: ON THE MINIMUM VOLUME COVERING ELLIPSOID OF ELLIPSOIDS Prop 2.6
# https://repository.bilkent.edu.tr/server/api/core/bitstreams/786c21eb-6b4e-44fb-a0f6-35371441960e/content
def max_q_over_ellipse_sdp(Q, A, b=None, gamma=None, c=None, r_sq=1, surface_only=False):
    nx = A.shape[0]
    b = np.zeros((nx, 1)) if b is None else b
    gamma = 0 if gamma is None else gamma
    c = np.zeros((nx, 1)) if c is None else c

    b = b if b.ndim == 2 else b[:, np.newaxis]
    c = c if c.ndim == 2 else c[:, np.newaxis]

    Q = Q if r_sq == 1 else Q / r_sq

    F = np.block([[A, b], [b.T, gamma]])
    G = np.block([[Q, -Q@c], [-c.T@Q, c.T@Q@c-1]])

    X = cp.Variable((nx+1, nx+1), symmetric=True)
    constraints = [X >> 0, X[-1, -1] == 1]
    if surface_only:
        constraints.append(cp.trace(G@X) == 0)
    else:
        constraints.append(cp.trace(G@X) <= 0)

    obj = cp.trace(F@X)
    prob = cp.Problem(cp.Maximize(obj), constraints)
    result = prob.solve(solver=cp.MOSEK)
    if prob.status == "optimal":
        return result, X.value[:-1, -1]
    else:
        raise Exception("solution not found")
    
def max_q_over_ellipse_sample(Q, A, c, r_sq, n_samples=1e5):
    n_samples = int(n_samples)
    d  = Q.shape[0]
    Q_sqrt = la.sqrtm(psd_inverse(Q, det=False))

    samples = np.random.rand(n_samples, d) * 2 - 1
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    samples = np.sqrt(r_sq) * samples @ Q_sqrt + c
    return np.max(samples[:, None] @ A @ samples[..., None])


def get_logger(path=None):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.addHandler(logging.FileHandler(path+".log", 'w'))
    return logger

def create_folder_if_not_exist(args, folder, logger=None):
    folder = os.path.join("outputs", folder)
    if os.path.exists(folder):
        for folder in glob.glob(os.path.join(folder, "*")):
            if not os.path.isdir(folder):
                continue
            args_path = os.path.join(folder, "args.pkl")
            if not os.path.exists(args_path):
                continue
            with open(args_path, "rb") as f:
                args_in_folder = pickle.load(f)
            if args_in_folder == args:
                logger = get_logger()
                logger.info(f"Found existing folder for current args: {folder}, delete it to re-run")
                return None
    str_time = datetime.now().strftime('%Y%m%d%H%M%S')
    dir_name = os.path.join(folder, str_time)
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, "args.pkl"), "wb") as f:
        pickle.dump(args, f)
    return dir_name

def get_last_folder(folder):
    folders = list(sorted(glob.glob(os.path.join("outputs", folder, "*"))))[::-1]
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        return folder
    return None

def compute_metrics(sys, us, bar_x0, Sigma_x0, Hy, costQ, costR):
    mus = [bar_x0]
    Sigmas = [Sigma_x0]
    for u in us:
        mus.append(sys.A @ mus[-1] + sys.B @ u)
        Sigmas.append(sys.A @ Sigmas[-1] @ sys.A.T + sys.E @ sys.Q @ sys.E.T)
    mus = np.array(mus)[1:] @ sys.C.T
    Sigmas = sys.C @ np.array(Sigmas)[1:] @ sys.C.T + sys.R
    exp_meas_cost = np.sum(np.squeeze(mus[:, None] @ costQ @ mus[..., None]) + np.trace(Sigmas @ costQ, axis1=-1, axis2=-2))
    inp_cost = np.sum(np.squeeze(us[:, None] @ costR @ us[..., None]))
    exp_cost = exp_meas_cost + inp_cost
    cc_sat = []
    for h in Hy:
        _mus = mus @ h
        _Sigmas = np.sqrt(h @ Sigmas @ h)
        cc_sat.append(norm.cdf((1-_mus) /_Sigmas))
    cc_sat = np.array(cc_sat)
    return exp_cost, cc_sat