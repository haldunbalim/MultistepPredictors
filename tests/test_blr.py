import numpy as np
from msp.id import compute_calGw, compute_sigma_banded
from msp.utils import bandedh2dense, compute_msp
import scipy.linalg as la
import scipy.sparse as sp
from d2pc import generate_random_lti, RandomNormalController, hankelize, block_diag
import unittest

class TestBLR(unittest.TestCase):
    def test_calGw(self, nx=4, ny=2, nu=2, T=500):
        sys = generate_random_lti(nx, ny, nu)
        _, ys, us = sys.simulate(T, RandomNormalController(nu))

        xposts, _, _, P_priors, _, es, _, Sinvs = sys.kf_fwd(
            ys, us, return_aux=True, use_steady=True)
        PC = P_priors @ sys.C.T
        Ls = PC @ Sinvs
        Ks = [1, 3, 5, 10]
        for K in Ks:
            G0u, Gw = compute_msp(sys, K)
            calGw = compute_calGw(Gw, K, Ls)

            I = np.concatenate([xposts[:-K], hankelize(us, K)], axis=-1)
            I = np.kron(I, np.eye(ny))
            O = ys[K-1:].flatten()
            self.assertTrue(np.allclose(O - I @ G0u.ravel("F"), calGw @ es.flatten()))