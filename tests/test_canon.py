import numpy as np
from d2pc import rand_psd, generate_random_lti, hankelize, RepController
from msp.utils import ss_to_arx_params
from msp.utils.canon_sys import *
import unittest

class TestCanon(unittest.TestCase):
    def test_canon_syss(self, order=3, ny=3, nu=2, T=250):
        order, ny, nu = 4, 3, 2
        sys = generate_random_lti(order*ny, ny, nu)
        As, Bs = ss_to_arx_params(sys.A, sys.B, sys.C)

        Q, R, P0 = rand_psd(ny), rand_psd(ny), rand_psd(order*ny, beta=1)
        mu0 = np.random.rand(order*ny)

        obs_canon_sys = ObsCanonSystem(As, Bs, Q, R, mu0, P0)
        canon_sys = CanonSystem(As, Bs, Q, R, mu0, P0)
        io_sys = IOSystem(As, Bs, Q, R, mu0, P0)

        _, ys, us = io_sys.simulate(T, RepController(io_sys.nu, [-1, 0, 1], 5))
        us_hank = hankelize(np.vstack([np.zeros((order-1, nu)), us]), order)
        ll_obs = np.sum(obs_canon_sys.kf_fwd(ys, us)[-1])
        ll_canon = np.sum(canon_sys.kf_fwd(ys, us_hank)[-1])
        ll_io = np.sum(io_sys.kf_fwd(ys, us)[-1])
        self.assertTrue(np.allclose(ll_obs, ll_canon) and np.allclose(ll_obs, ll_io))

    def test_conversions(self, order=3, ny=3, nu=2, T=500):
        sys = generate_random_lti(order*ny, ny, nu)
        As, Bs = ss_to_arx_params(sys.A, sys.B, sys.C)

        Q, R, P0 = rand_psd(ny), rand_psd(ny), rand_psd(order*ny, beta=1)
        mu0 = np.random.rand(order*ny)

        obs_canon_sys = ObsCanonSystem(As, Bs, Q, R, mu0, P0)
        canon_sys = CanonSystem(As, Bs, Q, R, mu0, P0)
        io_sys = IOSystem(As, Bs, Q, R, mu0, P0)

        _, ys, us = obs_canon_sys.simulate(
            T, RepController(io_sys.nu, [-1, 0, 1], 5))
        us_hank = hankelize(np.vstack([np.zeros((order-1, nu)), us]), order)
        ll_io = np.sum(io_sys.kf_fwd(ys, us)[-1])

        tmp = convert_canon_to_IO(canon_sys)
        self.assertTrue(np.allclose(ll_io, np.sum(tmp.kf_fwd(ys, us)[-1])))
        tmp = convert_obs_canon_to_IO(obs_canon_sys)
        self.assertTrue(np.allclose(ll_io, np.sum(tmp.kf_fwd(ys, us)[-1])))
        tmp = convert_canon_to_obs_canon(canon_sys)
        self.assertTrue(np.allclose(ll_io, np.sum(tmp.kf_fwd(ys, us)[-1])))
        tmp = convert_IO_to_obs_canon(io_sys)
        self.assertTrue(np.allclose(ll_io, np.sum(tmp.kf_fwd(ys, us)[-1])))
        tmp = convert_obs_canon_to_canon(obs_canon_sys)
        self.assertTrue(np.allclose(ll_io, np.sum(tmp.kf_fwd(ys, us_hank)[-1])))
        tmp = convert_IO_to_canon(io_sys)
        self.assertTrue(np.allclose(ll_io, np.sum(tmp.kf_fwd(ys, us_hank)[-1])))




