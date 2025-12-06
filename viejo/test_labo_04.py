import numpy as np
import pytest
from labo_04 import res_tri

def test_res_tri_lower_triangular():
    # L is lower triangular, solve Lx = b
    L = np.array([[2, 0, 0],
                  [3, 1, 0],
                  [1, -1, 1]], dtype=float)
    B = np.array([2, 5, 0], dtype=float)
    # The solution should be x = [1, 2, -1]
    X = res_tri(L, B, inferior=True)
    np.testing.assert_allclose(L @ X, B)
    np.testing.assert_allclose(X, [1, 2, -1])

def test_res_tri_upper_triangular():
    # U is upper triangular, solve Ux = b
    U = np.array([[2, 1, -1],
                  [0, 1, 2],
                  [0, 0, 3]], dtype=float)
    B = np.array([2, 4, 9], dtype=float)
    # The solution should be x = [1, 1, 3]
    X = res_tri(U, B, inferior=False)
    np.testing.assert_allclose(U @ X, B)
    np.testing.assert_allclose(X, [1, 1, 3])

def test_res_tri_identity():
    # Identity matrix, solution should be B
    I = np.eye(4)
    B = np.array([1, 2, 3, 4], dtype=float)
    X = res_tri(I, B, inferior=True)
    np.testing.assert_allclose(X, B)

def test_res_tri_zero_rhs():
    # Any triangular matrix with zero RHS should return zero vector
    L = np.tril(np.ones((3, 3)))
    B = np.zeros(3)
    X = res_tri(L, B, inferior=True)
    np.testing.assert_allclose(X, [0, 0, 0])

def test_res_tri_random_upper():
    np.random.seed(0)
    U = np.triu(np.random.rand(5, 5) + 1)  # ensure nonzero diagonal
    X_true = np.arange(1, 6)
    B = U @ X_true
    X = res_tri(U, B, inferior=False)
    np.testing.assert_allclose(X, X_true)