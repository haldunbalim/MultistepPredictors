import numpy as np
from msp.utils import bandedh2dense, dense2bandedh
import unittest

class TestMisc(unittest.TestCase):
    def test_banded2hdense(self, d=100, band_sz=5):
        A = np.random.rand(d, d)
        A = A.T @ A
        for i in range(band_sz, d):
            A[np.triu_indices(A.shape[0], k=i)] = 0
        for i in range(-d, -band_sz):
            A[np.tril_indices(A.shape[0], k=i+1)] = 0
        ans = bandedh2dense(dense2bandedh(A, band_sz, lower=True), lower=True)
        self.assertTrue(np.allclose(A, ans), "Failed to convert banded to dense with upper")
        bandedh2dense(dense2bandedh(A, band_sz, lower=False), lower=False)
        self.assertTrue(np.allclose(A, ans), "Failed to convert banded to dense with lower")
