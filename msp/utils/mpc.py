import numpy as np
import cvxpy as cp
from scipy.stats import norm, chi2
from d2pc import block_diag
from .misc import compute_msp
import scipy.linalg as la

def solve_nominal_mpc(sys, N, bar_x0, Sigma_x0, Hy, Hu, costQ, costR, p):
    # optimization for the true system
    ys = cp.Variable((N, sys.ny))
    us = cp.Variable((N, sys.nu))
    costQ_sqrt = la.sqrtm(costQ)
    costR_sqrt = la.sqrtm(costR)

    constraints = []
    for i in range(N):
        G0u, Gw = compute_msp(sys, i+1, use_E=True)
        G0 = G0u[:, :sys.nx]
        covy = G0 @ Sigma_x0 @ G0.T + Gw @ block_diag(sys.Q, i+1) @ Gw.T + sys.R
        tights = np.sqrt(np.array([h.T @ covy @ h for h in Hy]))
        constraints += [
            ys[i] == G0u @ cp.hstack([bar_x0, us[:i+1].flatten("C")]),
            Hy @ ys[i] <= 1 - norm.ppf(p) * tights,
            Hu @ us[i] <= 1
        ]

    cost = cp.sum_squares(ys @ costQ_sqrt)
    cost += cp.sum_squares(us @ costR_sqrt)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS)
    return us.value


def solve_msp_mpc(N, mus, covs, bar_x0, Hy, Hu, costQ, costR, delta, epsilon, tights, ellipsoidal=False):
    ny, nu = Hy.shape[1], Hu.shape[1]
    # optimization for the true system
    ys = cp.Variable((N, ny))
    us = cp.Variable((N, nu))
    costQ_sqrt = la.sqrtm(costQ)
    costR_sqrt = la.sqrtm(costR)
    cov_sqrts = [la.sqrtm(cov) for cov in covs]

    constraints = []
    coeff = norm.ppf(1 + delta - epsilon)
    for i in range(N):
        xu = cp.hstack([bar_x0, us[:i+1].flatten("C")])
        constraints += [
            ys[i] == mus[i].reshape(ny, -1, order="F") @ xu,
            Hu @ us[i] <= 1
        ]
        if ellipsoidal:
            ntheta = cov_sqrts[i].shape[0]
            coeff = np.sqrt(chi2.ppf(delta, ntheta))

        for hy, _tight in zip(Hy, tights[i]):
            cov_sqrt = cov_sqrts[i] @ np.kron(np.eye(xu.size), hy[:, None])
            __tight = coeff * cp.norm(cov_sqrt @ xu) + _tight
            constraints.append(hy @ ys[i] <= 1 - __tight)

    cost = cp.sum_squares(ys @ costQ_sqrt)
    cost += cp.sum_squares(us @ costR_sqrt)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS)
    if prob.status != cp.OPTIMAL:
        raise ValueError("Infeasible")
    return ys.value, us.value
