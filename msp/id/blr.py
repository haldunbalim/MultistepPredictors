from msp.utils import IOSystem
from d2pc import hankelize, psd_inverse
from .blr import *
from .blr_jk import *
from .blr_alt import *
import numpy as np
import scipy.linalg as la
from typing import Optional, List

def blr(sys:IOSystem, ys:np.ndarray, us:np.ndarray, Ks:List[int], mode:Optional[str]="kf"):
    if mode == "meas" and not isinstance(sys, IOSystem):
        raise ValueError("for meaurement based blr sys must be a IOSystem form")
    if mode == "kf":
        x_posts, _, _, P_priors, _, _, Ss, Sinvs = sys.kf_fwd(ys, us, return_aux=True, use_steady=True)
        Ls = P_priors @ sys.C.T @ Sinvs
        calS = sp.block_diag(Ss)
    elif mode == "kf-alt":
        x_posts, P_posts, _, es, Ss, Sinvs = sys.kf_fwd_alt(
            ys, us, return_aux=True, use_steady=True)
        PC = P_posts[1:] @ sys.C.T
        Ls = sys.A @ PC @ Sinvs
        calS = sp.block_diag(Ss)
    elif mode != "meas":
        raise ValueError("mode must be either 'kf', 'meas' or 'kf-alt'")
    ls = []
    for K in Ks:
        if mode == "kf":
            Sigma_bd = compute_sigma_banded(sys, K, Ls, calS)
            I = np.concatenate([x_posts[:-K], hankelize(us, K)], axis=-1)
            O = ys[K-1:].flatten()
        elif mode == "kf-alt":
            Sigma_bd = compute_sigma_banded_alt(sys, K, Ls, calS)
            I = np.concatenate([x_posts[1:-K], hankelize(us[1:], K)], axis=-1)
            O = ys[K:].flatten()
        elif mode == "meas":
            order, nu = sys.order, sys.nu
            Sigma_bd = compute_sigma_banded_jk(sys, K=K, T=len(ys))
            I = np.concatenate([hankelize(us[:-K], order)[:, nu:],
                                hankelize(ys[:-K], order),
                                hankelize(us[order:], K)], axis=1)
            O = ys[K+order-1:].flatten()

        I = np.kron(I, np.eye(sys.ny))
        IT_Sigma_inv = la.lapack.spbsv(Sigma_bd, I, lower=True)[1].T
        prec = IT_Sigma_inv @ I
        covar = psd_inverse(prec, det=False)
        cross_covar = IT_Sigma_inv @ O
        mu_est = covar @ cross_covar
        ls.append((mu_est, prec))
    return ls
