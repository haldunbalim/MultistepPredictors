import numpy as np
from scipy.stats import chi2, norm
from d2pc.utils import psd_inverse, block_diag, zero_mat
from msp.utils import GenChi2
from .misc import max_q_over_ellipse_sdp, compute_msp, max_q_over_ellipse_sample

# for true system
def compute_tight_true(sys, Ks, bar_x0, Sigma_x0, us, h, p, two_sided=False):
    tights_true = [h.T @ (sys.C @ Sigma_x0 @
                          sys.C.T + sys.R) @ h]
    mean_true = [sys.C @ bar_x0]
    for K in Ks:
        G0u, Gw = compute_msp(sys, K)
        G0, Gu = G0u[:, :sys.nx], G0u[:, sys.nx:]
        tights_true.append(h.T @ (G0 @ Sigma_x0 @ G0.T + Gw @ block_diag(
            sys.E @ sys.Q @ sys.E.T, K) @ Gw.T + sys.R) @ h)
        mean_true.append(G0 @ bar_x0 + Gu @ us[:K].flatten())

    coeff = np.sqrt(chi2.ppf(p, 1)) if two_sided else norm.ppf(p)
    tights_true = coeff * np.sqrt(np.array(tights_true))
    mean_true = np.array(mean_true) @ h
    return mean_true, tights_true

# computes tightening by sampling
def ct_sampling(sys, Ks, mus, covs, bar_x0, Sigma_x0, us, h, p, delta, n_samples=1e5, two_sided=False):
    n_samples = int(n_samples)
    mean_sampling = [h@sys.C@bar_x0]
    coeff = np.sqrt(chi2.ppf(p, 1)) if two_sided else norm.ppf(p)
    tights_sampling = [
        coeff * np.sqrt(h.T @ (sys.C @ Sigma_x0 @ sys.C.T + sys.R) @ h)]

    for K, mu, cov in zip(Ks, mus, covs):
        ths = np.random.multivariate_normal(mu, cov, n_samples)
        G0u = ths.reshape(n_samples, sys.ny, -1, order="F")
        G0 = G0u[:, :, :sys.nx]
        Gw = compute_msp(sys, K)[1]
        cov_off = Gw @ block_diag(sys.E @ sys.Q @
                                  sys.E.T, K) @ Gw.T + sys.R
        _coff = h.T @ (G0 @ Sigma_x0 @ np.swapaxes(G0, -1, -2) + cov_off) @ h

        z = np.hstack([bar_x0, us[:K].flatten()])
        if two_sided:
            vals = ths @ np.kron(z, h) + np.random.choice(
                [-1, 1], len(_coff)) * coeff * np.sqrt(_coff)
        else:
            vals = ths @ np.kron(z, h) + coeff * np.sqrt(_coff)
        mean_sampling.append(np.mean(vals))
        diffs = vals - mean_sampling[-1]
        if two_sided:
            diffs = np.abs(diffs)
        tights_sampling.append(np.percentile(diffs, delta*100))
    mean_sampling, tights_sampling = np.array(
        mean_sampling), np.array(tights_sampling)
    return mean_sampling, tights_sampling


# Köhler, Johannes, et al. 
# "State space models vs. multi-step predictors in predictive control: 
# Are state space models complicating safe data-driven designs?."
# Lemma 3
def compute_tights_msp_ellipsoidal(sys, Ks, mus, covs, bar_x0, Sigma_x0, us, h, p, delta, two_sided=False):
    cov_norms = [0]
    mean_est = [h.T @ sys.C @ bar_x0]
    coeffs = [0]
    for K, mu, cov in zip(Ks, mus, covs):
        xuh = np.kron(np.hstack([bar_x0, us[:K].flatten()]), h)
        mean_est.append(xuh.T @ mu)
        cov_norms.append(xuh.T @ cov @ xuh)
        nth = xuh.shape[0]
        coeff = np.sqrt(chi2.ppf(delta, nth)) if two_sided else np.sqrt(chi2.ppf(2*delta -1, nth))
        coeffs.append(coeff)

    mean_est = np.array(mean_est)
    tights_mean = np.array(coeffs) * np.sqrt(cov_norms)

    tights_cov = compute_tights_msp_ellipsoidal_offline(sys, Ks, mus, covs, Sigma_x0, h, p, delta, two_sided=False)
    return mean_est, tights_mean, tights_cov

def compute_tights_msp_ellipsoidal_offline(sys, Ks, mus, covs, Sigma_x0, Hy, p, delta, two_sided=False):
    tights = []
    Hy = np.atleast_2d(Hy)
    d = sys.nx * sys.ny
    for K, mu, cov in zip(Ks, mus, covs):
        _tights = []
        _, Gw = compute_msp(sys, K)
        for h in Hy:
            Ih = np.kron(np.eye(sys.nx), h)
            prec = psd_inverse(Ih @ cov[:d, :d] @ Ih.T, det=False)
            _mu = Ih @ mu[:d]
            try:
                tight, _ = max_q_over_ellipse_sdp(prec, Sigma_x0, b=None, gamma=None, c=_mu, r_sq=chi2.ppf(delta, cov.shape[0]))
            except:
                tight = max_q_over_ellipse_sample(prec, Sigma_x0, _mu, chi2.ppf(delta, cov.shape[0]), n_samples=1e5)
            c_jk = h.T @ (Gw @ block_diag(sys.E @ sys.Q @
                                          sys.E.T, K) @ Gw.T + sys.R) @ h
            _tights.append(tight + c_jk)
        tights.append(_tights)
    
    ptilde = p / delta
    coeff = np.sqrt(chi2.ppf(ptilde, 1)) if two_sided else norm.ppf(ptilde)
    tights = coeff * np.sqrt(np.array(tights))
    coeff0 = np.sqrt(chi2.ppf(p, 1)) if two_sided else norm.ppf(p)
    tight0 = coeff0 * np.sqrt(np.array([hy.T @ (sys.C @ Sigma_x0 @ sys.C.T + sys.R) @ hy for hy in Hy]))
    tights = np.vstack([[tight0], tights])
    return np.squeeze(tights)

# proposed solution
def compute_tights_msp(sys, Ks, mus, covs, bar_x0, Sigma_x0, us, h, p, delta, epsilon=None, two_sided=False):
    cov_norms = [0]
    mean_est = [h.T @ sys.C @ bar_x0]
    if epsilon is None:
        epsilon = (1+delta)/2

    for K, mu, cov in zip(Ks, mus, covs):
        xuh = np.kron(np.hstack([bar_x0, us[:K].flatten()]), h)
        mean_est.append(xuh.T @ mu)
        cov_norms.append(xuh.T @ cov @ xuh)

    mean_est = np.array(mean_est)
    coeff = np.sqrt(chi2.ppf(epsilon, 1)) if two_sided else norm.ppf(epsilon)
    tights_mean = coeff * np.sqrt(cov_norms)

    tights_cov = compute_tights_msp_offline(
        sys, Ks, mus, covs, Sigma_x0, h, p, delta, epsilon, two_sided=two_sided)
    return mean_est, tights_mean, tights_cov

def compute_tights_msp_offline(sys, Ks, mus, covs, Sigma_x0, Hy, p, delta, epsilon, two_sided=False):
    tights = []
    Hy = np.atleast_2d(Hy)
    for K, mu, cov in zip(Ks, mus, covs):
        _tights = []
        _, Gw = compute_msp(sys, K)
        for h in Hy:
            M = np.kron(block_diag(Sigma_x0, zero_mat(K*sys.nu)), np.outer(h, h))
            genchi = GenChi2.from_quad(mu, cov, M)
            c_jk = h.T @ (Gw @ block_diag(sys.E @ sys.Q @ sys.E.T, K) @ Gw.T + sys.R) @ h
            try:
                gchi = genchi.ppf(1+delta-epsilon)
            except:
                gchi = genchi.ppf_sample(1+delta-epsilon, n_samples=1e6)
            _tights.append(gchi + c_jk)    
        tights.append(_tights)
    
    ptilde = p / delta
    coeff = np.sqrt(chi2.ppf(ptilde, 1)) if two_sided else norm.ppf(ptilde)
    tights = coeff * np.sqrt(np.array(tights))
    coeff0 = np.sqrt(chi2.ppf(p, 1)) if two_sided else norm.ppf(p)
    tight0 = coeff0 * np.sqrt(np.array([hy.T @ (sys.C @ Sigma_x0 @ sys.C.T + sys.R) @ hy for hy in Hy]))
    tights = np.vstack([[tight0], tights])
    return np.squeeze(tights)