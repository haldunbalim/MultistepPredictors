import numpy as np
from d2pc.id_em import estimate_arx_sys, LTISystem
import control as ct
import numpy as np
import scipy.linalg as la
from d2pc.utils import psd_inverse, block_diag
from d2pc.experiment import fwd_reach_nom_tube
from scipy.stats import norm, chi2
from msp.utils import CanonSystem, IOSystem, ss_to_arx_params
from typing import Optional 
from d2pc import (create_spring_mass_sys_ct, RSMPCController, timer, nom_tube_design, 
                  compute_nom_tight, compute_stoch_tight, stoch_tube_design_time_varying, 
                  get_nll_hess, DynOFController)

def estimate_arx_sys_canon(ys: np.ndarray, us: np.ndarray, order: int,
                           Q: Optional[np.ndarray] = None, Qtype: Optional[str] = "full",
                           R: Optional[np.ndarray] = None, Rtype: Optional[str] = "full",
                           P0: Optional[np.ndarray] = None, warm_start_ls: bool = True,
                           ignore_Theta: bool = False, **em_kwargs):
    syss, Es = estimate_arx_sys(ys, us, order, Q=Q, Qtype=Qtype, R=R, Rtype=Rtype, P0=P0, warm_start_ls=warm_start_ls,
                                ignore_Theta=ignore_Theta, convert_to_std=False,**em_kwargs)

    if "path" in em_kwargs and em_kwargs["path"]:
        syss = [CanonSystem.from_canon_lti(sys) for sys in syss]
        return syss, Es
    else:
        syss = CanonSystem.from_canon_lti(syss)
        return syss, Es
    

def compute_tight_d2pc(ctrlr, u_ref, p, h, x0: Optional[np.ndarray] = None, two_sided=False):
    """
        Setup the data for the S-DDPC comparison for ARX models
        - ctrlr: the controller
        - u_ref: the reference us
        - p: the probability of chance constraint satisfaction
        - h: constraint vector
    """
    sys = ctrlr.sys
    if x0 is None:
        x0 = np.zeros(sys.nx)
    coeff = np.sqrt(chi2.ppf(p, 1)) if two_sided else norm.ppf(p)
    stoch_tights = np.array([coeff * np.sqrt(h.T @ (sys.C @ calX[:sys.nx,
                            :sys.nx] @ sys.C.T + sys.R) @ h) for calX in ctrlr.sigma_tvs])
    stoch_tights = np.concatenate(
        [stoch_tights, [stoch_tights[-1]] * (len(u_ref)-len(stoch_tights)+1)])

    calA, calBv = ctrlr.dynof.calA, ctrlr.dynof.calBv
    nus = []
    xis = [np.concatenate([x0, np.zeros_like(x0)])]
    for u in u_ref:
        nus.append(u - ctrlr.dynof.K @ xis[-1][sys.nx:])
        xis.append(calA @ xis[-1] + calBv @ nus[-1])
    nus = np.array(nus)
    xis = np.array(xis)
    tubes_lmi, _, _ = fwd_reach_nom_tube(ctrlr, nus, x0=x0)

    Pinv = psd_inverse(ctrlr.calP)
    f = np.sqrt(h.T @ sys.C @
                Pinv[:sys.nx, :sys.nx] @ sys.C.T @ h)
    means = xis[:, :sys.nx] @ sys.C.T @ h
    return means, f * tubes_lmi, stoch_tights

def construct_spring_mass_sys(n_m, n_a, q, r):
    # construct true system
    Ac, Bc, J = create_spring_mass_sys_ct(
        [1]*n_m, [10]*n_m, [2]*n_m, num_actuated=n_a)
    C = np.eye(n_m*2)[::2]
    ss = ct.c2d(ct.ss(Ac, Bc, C, 0), .5)
    As, Bs = ss_to_arx_params(ss.A, ss.B, ss.C)
    mu0 = np.zeros(n_m*2)
    P0 = np.eye(n_m*2)
    Q = np.eye(n_m)*q
    R = np.eye(n_m)*r
    sys = IOSystem(As, Bs, Q, R, mu0, P0)
    sys.P0 = sys.get_steady_post_covar()
    return sys


def d2pc_pipeline_open_loop(est_sys: LTISystem, ys:np.ndarray, us:np.ndarray, J: np.ndarray, th0: np.ndarray,
                            p_cc: float, delta: float, costQ: np.ndarray, costR: np.ndarray, H: np.ndarray,
                            init_covar: np.ndarray, T_err: int, logger= None):
    
    """
        Pipeline for the D2PC controller synthesis
        - est_sys: LTISystem - estimated system
        - ys: np.ndarray - output data
        - us: np.ndarray - input data
        - J: np.ndarray - affine parametrization matrix
        - th0: np.ndarray - affine parametrization constant
        - p_cc: float - chance constraint satisfaction prob
        - delta: float - probability that the true system is covered
        - costQ: np.ndarray - cost matrix for the disturbance cov
        - costR: np.ndarray - cost matrix for the measurement noise cov
        - H: np.ndarray - state-input constraints
        - init_covar: np.ndarray - initial covariance of the state
        - T_err: int - error covariance computation horizon
    """
    with timer() as t_off:
        # quantify the parameteric uncertainty
        with timer() as t:
            hess_th = get_nll_hess(est_sys, J, th0, ys, us)
            if logger is not None:
                logger.info("Uncertainty Quantification time: " + f"{t():.3f}"+"s")

        # robust output-feedback controller synthesis
        with timer() as t:
            # set robust output-feedback controller to 0 to make it open loop
            Ac = np.zeros((est_sys.nx, est_sys.nx))
            K = np.zeros((est_sys.nu, est_sys.nx))
            L = np.zeros((est_sys.nx, est_sys.ny))
            dynof = DynOFController(est_sys, Ac, K, L)
            if logger is not None:
                logger.info("Output-feedback controller synthesis time: " +
                            f"{t():.3f}"+"s")

        # offline design of the controller
        with timer() as t:
            ctrlr = RSMPCController(est_sys, dynof, H, p_cc, J, hess_th, delta, costQ, costR)
            # nominal tube design
            _, ctrlr.rho, ctrlr.calP = nom_tube_design(est_sys, dynof, H, J, hess_th, delta)
            if logger is not None:
                logger.info("Nominal tube design time: " + f"{t():.3f}"+"s")
                
        with timer() as t:
            # time-varying stoch tightening
            init_xi_covar = block_diag(init_covar, np.zeros_like(init_covar))
            assert isinstance(T_err, int) and T_err >= 1, "T_err must be a positive integer"
            T_err = 1 if init_xi_covar is None else T_err
            _, sigma_tvs = stoch_tube_design_time_varying(est_sys, dynof, T_err, init_xi_covar, H, J, hess_th, delta)
            # add R to sigma for the measurement constraints
            Cpinv = la.pinv(est_sys.C)
            sigma_tvs += block_diag(Cpinv @ est_sys.R @ Cpinv.T, np.zeros((est_sys.nx, est_sys.nx)))
            ctrlr.sigma_tvs = sigma_tvs
            if logger is not None:
                logger.info("Time-varying stochastic tube design time: " + f"{t():.3f}"+"s")

        ctrlr.nom_tight = compute_nom_tight(ctrlr.calP, ctrlr.dynof.K, ctrlr.H)
        ctrlr.stoch_tight_tv = np.array([compute_stoch_tight(sigma_t, ctrlr.dynof.K,
                                                            ctrlr.H, ctrlr.p_cc)
                                        for sigma_t in ctrlr.sigma_tvs])
        ctrlr.Sc = block_diag(ctrlr.costQ, np.zeros((est_sys.nx, est_sys.nx)))

        if logger is not None:
            logger.info("Total offline time: " + f"{t_off():.3f}"+"s")
    return ctrlr