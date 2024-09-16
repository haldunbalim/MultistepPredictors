import warnings
import numpy as np
import scipy.integrate as integrate
from scipy.stats import ncx2, chi2, norm
from scipy.optimize import fsolve, brentq
import scipy.linalg as la
from d2pc.utils import symmetrize

# https://arxiv.org/abs/2012.14331
# https://www.mathworks.com/matlabcentral/fileexchange/85028-generalized-chi-square-distribution

class GenChi2:
    def __init__(self, w, k, lambda_, m, s):
        self.w = w
        self.k = k
        self.lambda_ = lambda_
        self.m = m
        self.s = s

    # Computes Prob[x<Q] for generalized chi-squared distribution
    def cdf(self, Q, method="imhof", **kwargs):
        if method == "ruben":
            return self.cdf_ruben(Q, **kwargs)
        elif method == "imhof":
            return self.cdf_imhof(Q, **kwargs)
        else:
            raise ValueError("Invalid method")
        
    def cdf_imhof(self, Q, ub=np.inf, limit=200):
        def imhof_integrand(u, x, w, k, lambda_, s):
            theta = np.sum(k * np.arctan(w * u) + (lambda_ *
                           (w * u)) / (1 + w**2 * u**2)) / 2 - u * x / 2
            rho = np.prod((1 + w**2 * u**2)**(k / 4) * np.exp(((w**2 * u**2)
                          * lambda_) / (2 * (1 + w**2 * u**2)))) * np.exp(u**2 * s**2 / 8)
            f = np.sin(theta) / (u * rho)
            return f
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Warning: Integration may fail when upper bound is too large / limit is too small
            # need to adjust ub and limit accordingly
            res, _ = integrate.quad(lambda u: imhof_integrand(
                u, Q-self.m, self.w, self.k, self.lambda_, self.s,), 0, ub,
                epsabs=1e-10, epsrel=1e-2, limit=limit, full_output=False)
        return np.clip(0.5-res/np.pi, 0, 1)

    def cdf_ruben(self, Q, n_ruben=1e4):
        n_ruben = int(n_ruben)
        beta = 0.90625 * np.min(self.w)
        M = np.sum(self.k)
        n = np.arange(1, n_ruben)

        g1 = np.sum(self.k[:, None] * np.power((1-beta / self.w)
                    [:, None], n[None]), axis=0)
        g2 = beta * (self.lambda_/self.w) @ (n *
                                            np.power((1-beta / self.w)[:, None], n[None]-1))
        g = g1 + g2
        a = np.zeros(n_ruben)
        a[0] = np.sqrt(np.exp(-np.sum(self.lambda_)) * (beta**M)
                    * np.prod(1/np.power(self.w, self.k)))

        for i in range(n_ruben-1):
            a[i+1] = np.dot(np.flip(g[:i+1]), a[:i+1]) / 2 / (i+1)
        F = chi2.cdf((Q-self.m) / beta, np.arange(M, M+2*(n_ruben), 2))
        return np.clip(a.T @ F, 0, 1)
    
    def ppf(self, p, method_cdf="imhof", verbose=0, **kwargs):
        if p == 0:
            return 0
        unique_w = np.unique(self.w)
        if self.s==0 and len(self.w) == 1:
            w = unique_w[0]
            p = p if w>0 else 1-p
            return ncx2(sum(self.k), sum(self.lambda_)).ppf(p) * self.w + self.m
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.all(self.w >= 0), "This method is only valid for w >= 0 bcz of lb"
            lb, ub = self.ppf_lb(p), self.ppf_ub(p)
            try:
                Q, res = brentq(
                    lambda Q: self.cdf(Q, method=method_cdf, **kwargs)-p, 
                    lb, ub, full_output=True)
            except:
                if verbose > 0:
                    print("brentq failed, using fsolve")
                return self.ppf(p, method_solve="fsolve", method_cdf=method_cdf, **kwargs)
            if not res.converged:
                raise Exception(f"Optimization failed:")
        return Q
    
    def ppf_sample(self, p, n_samples=1e6):
        n_samples = int(n_samples)
        rng = np.random.default_rng()
        sampled_ncx2 = np.array([w*rng.noncentral_chisquare(k, lambda_, size=n_samples) 
                                 for k, lambda_, w in zip(self.k, self.lambda_, self.w) 
                                 for _ in range(k)])
        sampled_m = np.random.normal(self.m, self.s, size=n_samples)
        sampled_gxc = np.sum(sampled_ncx2, axis=0) + sampled_m
        return np.percentile(sampled_gxc, p*100)

    # ignore (th-\mu).T @ (th-\mu) 
    def ppf_lb(self, p):
        if not np.all(self.w >= 0):
            raise ValueError("This lower bound is only valid for w >= 0")
        mu_bar = np.array([np.sqrt(w) * np.sqrt(lambda_)
                        for k, lambda_, w in zip(self.k, self.lambda_, self.w)
                        for _ in range(k)] + [self.m])
        Sigma_bar = np.diag([w for k, w in zip(self.k, self.w)
                            for _ in range(k)] + [self.s**2])
        lb = mu_bar.T @ mu_bar + 4 * mu_bar.T @ Sigma_bar @ mu_bar * norm.ppf(p)
        return lb

    # use boole's inequality to upper bound ppf
    def ppf_ub(self, p):
        if np.allclose(self.s, 0):
            p = 1 - ((1 - p) / len(self.w))
        else:
            p = 1 - ((1 - p) / (len(self.w) + 1))
        ub = sum([ncx2(k, lambda_).ppf(p) * w
                for k, lambda_, w in zip(self.k, self.lambda_, self.w)]) + self.m
        if not np.allclose(self.s, 0):
            ub += self.s * norm.ppf(p)
        return ub

    @property
    def mu(self):
        return self.w.T @ (self.k + self.lambda_) + self.m
    
    @property
    def var(self):
        return 2*((self.w**2).T @ (self.k + 2*self.lambda_)) + self.s**2

    def __repr__(self):
        return f"GenChi2(w={self.w}, k={self.k}, lambda_={self.lambda_}, m={self.m:.5f}, s={self.s:.5f})"

    @staticmethod
    def from_quad(mu, Sigma, M, project=True):
        # Projects a quadratic form to non-zero subspace
        # This is suggested to accelerate the cdf / ppf computation
        def project_ss(mu, cov, M, tol=1e-12):
            d, R = la.eigh(M)
            cond = np.abs(d) > tol
            dp = d[cond]
            Rp = R[:, cond]
            Mp = np.diag(dp)
            assert np.allclose(M, Rp @ Mp @ Rp.T), "Reconstruction failed after projection"
            return Rp.T @ mu, Rp.T @ cov @ Rp, Mp
        if project:
            return GenChi2._from_quad(*project_ss(mu, Sigma, M))
        else:
            return GenChi2._from_quad(mu, Sigma, M)

    # Computes parameters of generalized chi-squared distribution params from quad form
    @staticmethod
    def _from_quad(mu, Sigma, A, b=None, c=0):
        A = symmetrize(A)  # symmetrize A
        b = b if b is not None else np.zeros(mu.shape)
        Sigma_sqrt = la.sqrtm(Sigma)
        # transform to z ~ N(0, I)
        A_z = Sigma_sqrt @ A @ Sigma_sqrt
        b_z = Sigma_sqrt @ (2*A@mu + b)
        c_z = mu.T @ A @ mu + b.T @ mu + c

        # transform to y ~ N(0, I)
        d, R = la.eigh(A_z)
        b_y = R.T @ b_z

        d_nz = d[d != 0]
        d_nz_unique = np.unique(d_nz)
        # Compute params
        w = d_nz_unique
        lambda_ = np.array([np.sum(b_y[d == el]**2)
                           for el in w]) / (4*w**2)
        k = np.array([np.sum(d_nz == el) for el in w])
        m = c_z - w.T @ lambda_
        s = np.linalg.norm(b_y[d == 0])
        return GenChi2(w, k, lambda_, m, s)
