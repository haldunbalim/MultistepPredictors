import numpy as np
from msp.id import compute_calGw_alt, compute_sigma_banded_alt
from msp.utils import bandedh2dense, compute_msp
import scipy.linalg as la
import scipy.sparse as sp
from d2pc import generate_random_lti, RandomNormalController, hankelize, block_diag
import unittest

class TestBLRAlt(unittest.TestCase):
    def test_calGw(self, nx=4, ny=2, nu=2, T=500):
        sys = generate_random_lti(nx, ny, nu)
        _, ys, us = sys.simulate(T, RandomNormalController(nu))

        x_posts, P_posts, _, es, _, Sinvs = sys.kf_fwd_alt(
            ys, us, return_aux=True, use_steady=True)
        PC = P_posts[1:] @ sys.C.T
        Ls = sys.A @ PC @ Sinvs
        Ks = [1, 3, 5, 10]
        for K in Ks:
            G0u, Gw = compute_msp(sys, K)
            calGw = compute_calGw_alt(Gw, K, Ls)

            I = np.concatenate([x_posts[1:-K], hankelize(us[1:], K)], axis=-1)
            I = np.kron(I, np.eye(ny))
            O = ys[K:].flatten()
            self.assertTrue(np.allclose(O - I @ G0u.ravel("F"), calGw @ es.flatten()))