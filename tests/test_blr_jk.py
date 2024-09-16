import numpy as np
from msp.id import compute_calGw_jk, compute_calGv_jk
from msp.utils import ss_to_arx_params, compute_msp, IOSystem
import scipy.linalg as la
from d2pc import generate_random_lti, RandomNormalController, hankelize, inv_tril, lchol
import unittest

class TestBLR(unittest.TestCase):
    def test_calGws(self, order=4, ny=3, nu=2, T=250):
        q, r, p0 = 1e-2, 1e-2, 1
        rand_sys = generate_random_lti(order*ny, ny, nu)
        As, Bs = ss_to_arx_params(rand_sys.A, rand_sys.B, rand_sys.C)
        Q = q * np.eye(ny)
        R = r * np.eye(ny)
        mu0 = np.zeros(order*ny)
        P0 = p0 * np.eye(order*ny)
        sys = IOSystem(As, Bs, Q, R, mu0, P0)

        # ctrlr = StaticController(np.zeros((T, nu)))
        ctrlr = RandomNormalController(nu)
        _, ys, us, ws, vs = sys.simulate(T, ctrlr, return_noises=True)
        Q_lchol, R_lchol = lchol(sys.Q), lchol(sys.R)
        Q_lchol_inv, R_lchol_inv = inv_tril(Q_lchol), inv_tril(R_lchol)
        ws, vs = ws @ Q_lchol_inv.T, vs @ R_lchol_inv.T

        Ks = [1, 3, 5, 10]
        for K in Ks:
            G0u, Gw = compute_msp(sys, K, use_E=True)
            G0 = G0u[:, :order*ny+order*nu-nu]

            calGw = compute_calGw_jk(Gw, order, T, K, Q_lchol=Q_lchol)
            calGv = compute_calGv_jk(G0, order, T, K, R_lchol=R_lchol)


            I = np.concatenate([hankelize(us[:-K], order)[:, nu:],
                                hankelize(ys[:-K], order),
                                hankelize(us[order:], K)], axis=1)
            
            I = np.kron(I, np.eye(ny))
            O = ys[K+order-1:].flatten()
            noise = calGw @ ws[order:].flatten() + calGv @ vs.flatten()
            self.assertTrue(np.allclose(O - I @ G0u.ravel("F"), noise))
