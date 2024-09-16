import numpy as np
import cvxpy as cp
from d2pc.id_em import LTISystem
from .misc import shifter
from d2pc.utils import zero_mat, block_diag, symmetrize
import scipy.linalg as la

class MSPSystem(LTISystem):
    pass

class ObsCanonSystem(MSPSystem):
    def __init__(self, As: np.ndarray, Bs: np.ndarray, Q: np.ndarray, R: np.ndarray, mu0: np.ndarray, P0: np.ndarray):
        """
            - As: np.ndarray (order, ny, ny)
            - Bs: np.ndarray (order, ny, nu)
            - mu0: np.ndarray (nx,) init mean of first outputs
            - P0: np.ndarray (nx, nx) init covariance of first outputs
        """
        self.As = As
        self.Bs = Bs
        order, ny = As.shape[:2]
        nx = order*ny
        tf = init_state_tf_canon(As)
        A = np.hstack([np.vstack([np.zeros((ny, nx-ny)), np.eye(nx-ny)]), np.vstack(As)])
        B = np.vstack(Bs)
        C = np.hstack([np.zeros((ny, nx-ny)), np.eye(ny)])
        mu0 = tf @ mu0
        P0 = symmetrize(tf @ P0 @ tf.T)
        super().__init__(A=A, B=B, C=C, E=C.T, Q=Q, R=R, mu0=mu0, P0=P0)

    @staticmethod
    def from_obs_canon_lti(sys: LTISystem):
        order = sys.nx // sys.ny
        As = np.array(np.vsplit(sys.A[:, -sys.ny:], order))
        Bs = np.array(np.vsplit(sys.B, order))
        return CanonSystem(As, Bs, sys.Q, sys.R, sys.mu0, sys.P0)
    

class CanonSystem(MSPSystem):
    def __init__(self, As: np.ndarray, Bs: np.ndarray, Q: np.ndarray, R: np.ndarray, mu0: np.ndarray, P0: np.ndarray):
        """
            - As: np.ndarray (order, ny, ny)
            - Bs: np.ndarray (order, ny, nu)
        """
        self.As = As
        self.Bs = Bs
        ny = As.shape[1]
        nx = As.shape[0] * ny
        A = np.vstack([np.hstack([np.zeros((nx-ny, ny)), np.eye(nx-ny)]), np.hstack(As)])
        B = np.vstack([np.zeros((nx-ny, Bs.shape[0] * Bs.shape[2])), np.hstack(Bs)])
        C = np.hstack([np.zeros((ny, nx-ny)), np.eye(ny)])
        super().__init__(A=A, B=B, C=C, E=C.T, Q=Q, R=R, mu0=mu0, P0=P0)

    @staticmethod
    def from_canon_lti(sys: LTISystem):
        order = sys.nx // sys.ny
        As = np.array(np.hsplit(sys.A[-sys.ny:], order))
        Bs = np.array(np.hsplit(sys.B[-sys.ny:], order))
        return CanonSystem(As, Bs, sys.Q, sys.R, sys.mu0, sys.P0)
    

class IOSystem(MSPSystem):
    def __init__(self, As: np.ndarray, Bs: np.ndarray, Q: np.ndarray, R: np.ndarray, mu0: np.ndarray, P0: np.ndarray):
        """
            - As: np.ndarray (order, ny, ny)
            - Bs: np.ndarray (order, ny, nu)
        """
        self.As = As
        self.Bs = Bs 
        order, ny, nu = Bs.shape

        Abot = np.hstack([np.hstack(Bs[:-1]), np.hstack(As)])
        Au = np.vstack([shifter(nu, order-1), np.zeros((nu, order*nu-nu))])
        Ay = shifter(ny, order)
        A = np.vstack([block_diag(Au, Ay), Abot])
        B = np.zeros((A.shape[0], nu))
        B[(order-2)*nu:order*nu-nu] = np.eye(nu)
        B[-ny:] = Bs[-1]

        C = np.hstack([np.zeros((ny, order*nu-nu)), np.zeros((ny, order*ny-ny)), np.eye(ny)])
        mu0 = np.hstack([np.zeros(order*nu-nu), mu0])
        P0 = block_diag(zero_mat(order*nu-nu), P0)

        super().__init__(A=A, B=B, C=C, E=C.T, Q=Q, R=R, mu0=mu0, P0=P0)

    @property
    def order(self):
        return (self.nx + self.nu) // (self.ny + self.nu)

    @staticmethod
    def from_IO_lti(sys: LTISystem):
        order = sys.nx // sys.ny
        As = np.array(np.hsplit(sys.A[-sys.ny:, -sys.ny*order:], order))
        Bs = np.concatenate([np.hsplit(sys.A[-sys.ny:, :-sys.ny*order], order-1) + [sys.B[-sys.ny:]]], axis=0)
        return CanonSystem(As, Bs, sys.Q, sys.R, sys.mu0, sys.P0)


def init_state_tf_canon(As):
    order = len(As)
    ny = As[0].shape[0]
    init_st_tf = np.zeros((order-1, ny, order-1, ny))
    for i in range(order-1):
        t = 0
        for j in range(order-i-2, order-1):
            init_st_tf[i, :, j, :] = As[t]
            t += 1
    init_st_tf = init_st_tf.reshape((order-1)*ny, (order-1)*ny)
    init_st_tf = block_diag(init_st_tf, np.eye(ny))
    return init_st_tf


def convert_IO_to_canon(io_sys: IOSystem):
    """
        This is the way I estimate systems is d2pc: state is past ys and input is hankelized
    """

    nx, ny, nu = io_sys.nx, io_sys.ny, io_sys.nu
    order = int((nx+nu) / (ny+nu))
    As = np.array(np.hsplit(io_sys.A[-ny:, -order*ny:], order))
    Bs = np.array(np.hsplit(io_sys.A[-ny:, :-order*ny], order-1) + [io_sys.B[-ny:]])
    mu0 = io_sys.mu0[-order*ny:]
    P0 = io_sys.P0[-order*ny:, -order*ny:]
    return CanonSystem(As, Bs, io_sys.Q, io_sys.R, mu0, P0)


def convert_obs_canon_to_canon(obs_canon: ObsCanonSystem):
    """
        Convert a obsv canonical system to d2pc form
    """
    ny = obs_canon.ny
    order = obs_canon.nx // ny
    As = np.array(np.vsplit(obs_canon.A[:, -ny:], order))
    Bs = np.array(np.vsplit(obs_canon.B, order))

    init_st_tf = init_state_tf_canon(As)
    init_st_tf = la.inv(init_st_tf)

    mu0 = init_st_tf @ obs_canon.mu0
    P0 = symmetrize(init_st_tf @ obs_canon.P0 @ init_st_tf.T)

    return CanonSystem(As, Bs, obs_canon.Q, obs_canon.R, mu0, P0)


def convert_canon_to_obs_canon(canon: CanonSystem):
    """
        Convert an d2pc form system to a observer canonical from state-space system
        inverse of: convert_back_canon
    """
    nx, ny, nu = canon.nx, canon.ny, canon.nu
    order = nx // ny
    nu = nu // order

    As = np.array(np.hsplit(canon.A[-ny:], order))
    Bs = np.array(np.hsplit(canon.B[-ny:], order))

    sys = ObsCanonSystem(As, Bs, Q=canon.Q, R=canon.R, mu0=canon.mu0, P0=canon.P0)
    return sys


def convert_canon_to_IO(canon: CanonSystem):
    """
        Convert a d2pc system to an input-output canonical form where state is past inputs+outputs
    """
    nx, ny, nu = canon.nx, canon.ny, canon.nu
    order = nx // ny
    nu = nu // order

    As = np.array(np.hsplit(canon.A[-ny:], order))
    Bs = np.array(np.hsplit(canon.B[-ny:], order))

    return IOSystem(As, Bs, canon.Q, canon.R, canon.mu0, canon.P0)


def convert_obs_canon_to_IO(obs_canon: ObsCanonSystem):
    return convert_canon_to_IO(convert_obs_canon_to_canon(obs_canon))


def convert_IO_to_obs_canon(io_sys: IOSystem):
    return convert_canon_to_obs_canon(convert_IO_to_canon(io_sys))
