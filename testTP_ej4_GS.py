import numpy as np
import scipy 
import inspect
from entrega import alc 

def is_QR_decomposition(A, Q, R, tol=1e-5):
    if not np.allclose(A, Q @ R, atol=tol):
        return False
    if not np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol):
        return False
    if not np.allclose(R, np.triu(R), atol=tol):
        return False
    return True

def pInvGramSchmidt_wrapper(X, Y):
    fun_params = get_arg_names(alc.pinvGramSchmidt)
    params = {"Y": Y}
    if "Q" in fun_params and "R" in fun_params:
        
        #Q, R = np.linalg.qr(X.T, mode='reduced')
        Q, R = alc.QR_con_GS(X.T)
         # Calculemos la reducida
        n = X.T.shape[1]
        Q = Q[:, :n]            
        R = R[:n, :n]
        params["Q"] = Q
        params["R"] = R
    return alc.pinvGramSchmidt(**params)

def pInvGramSchmidt_test_identity():
    X = np.eye(3)
    for _ in range(100):
        Y = np.random.randn(3, 3)
        W = pInvGramSchmidt_wrapper(X, Y)
        Y_pred = W @ X
        assert Y_pred.shape == Y.shape, f"Y_pred shape incorrect for identity matrix. Expected {Y.shape}, got {Y_pred.shape}"
        assert np.allclose(Y_pred, Y), "pInvGramSchmidt failed on identity matrix"

def pInvGramSchmidt_test_identity_wide_zero_error():
    X = np.eye(5,7)
    for _ in range(500):
        W_true = np.random.randn(32, 5)
        Y = W_true @ X  # esta en la imagen de X, asi que el error deberia poder ser 0
        W = pInvGramSchmidt_wrapper(X, Y)
        Y_pred = W @ X
        assert Y_pred.shape == Y.shape, f"Y_pred shape incorrect for tall identity matrix. Expected {Y.shape}, got {Y_pred.shape}"
        assert np.allclose(Y_pred, Y), "pInvGramSchmidt failed on tall identity matrix"
        assert np.allclose(W, W_true), "pInvGramSchmidt did not recover the correct weights for tall identity matrix"

def pInvGramSchmidt_test_identity_wide_known_error():
    # el truco aca es conseguir una base del espacio ortogonal de XT
    X = np.eye(5,7)
    for _ in range(500):
        # Yt = Xt Wt + E, con E en el espacio ortogonal a la imagen de Xt. Por la forma de Xt, lo sacamos facil
        Q_null = np.zeros((7,2))
        Q_null[-1,0] = 1
        Q_null[-2,1] = 1
 
        E = np.random.randn(32, Q_null.shape[1]) @ Q_null.T  # ruido en el espacio ortogonal
        W_true = np.random.randn(32, 5)
        Y = W_true @ X + E
        W = pInvGramSchmidt_wrapper(X, Y)
        Y_pred = W @ X
        error = Y - Y_pred
        error = np.linalg.norm(error, ord='fro')
        error_expected = np.linalg.norm(E, ord='fro')
        assert Y_pred.shape == Y.shape, f"Y_pred shape incorrect for tall identity matrix with known error. Expected {Y.shape}, got {Y_pred.shape}"
        assert np.allclose(error, error_expected, atol=1e-5), f"pInvGramSchmidt did not achieve the expected error {error_expected} for tall identity matrix with known error, got {error}"
        assert np.allclose(W, W_true, atol=1e-5), "pInvGramSchmidt did not recover the correct weights for tall identity matrix with known error"

def pInvGramSchmidt_test_random_wide_known_error():
    for _ in range(500):
        X = np.random.randn(5,7) * 100 # para numeros mas grandes 
        # no tan facil como antes pero usando la SVD de numpy se puede. Dado que X es aleatoria, el rango deberia ser completo
        U, _, _ = np.linalg.svd(X.T, full_matrices=True)
        Q_null = U[:, 5:]  # base del espacio nulo de XT

        E = 100 * np.random.randn(32, Q_null.shape[1]) @ Q_null.T  # ruido en el espacio ortogonal
        W_true = np.random.randn(32, 5)
        Y = W_true @ X + E

        W = pInvGramSchmidt_wrapper(X, Y)

        Y_pred = W @ X
        error = Y - Y_pred
        error = np.linalg.norm(error, ord='fro')
        error_expected = np.linalg.norm(E, ord='fro')
        assert Y_pred.shape == Y.shape, f"Y_pred shape incorrect for tall random matrix with known error. Expected {Y.shape}, got {Y_pred.shape}"
        assert np.allclose(error, error_expected, atol=1e-5), f"pInvGramSchmidt did not achieve the expected error {error_expected} for tall random matrix with known error, got {error}"
        assert np.allclose(W, W_true, atol=1e-5), "pInvGramSchmidt did not recover the correct weights for tall random matrix with known error"

def pInvGramSchmidt_test_solves_normal_eq():
    ns = np.arange(2,9)
    ps = np.arange(8, 16)
    m = 2
    sizes = np.array(list(zip(np.repeat(ns,len(ns)), np.tile(ps, len(ps)))))
    for n, p in sizes:   
        X = np.random.randn(n,p)
        Y = np.random.rand(m,p)
        W = pInvGramSchmidt_wrapper(X, Y)
        assert W.shape == (m,n),  f"W shape incorrect. Expected {(m,n)}, got {W.shape}"
        assert np.allclose((X@X.T)@W.T - (X@Y.T), np.zeros_like(W.T), atol=1e-4), "pInvGramSchmidt weights do not satisfy normal equations with 1e-4 tol"
        assert np.allclose((X@X.T)@W.T - (X@Y.T), np.zeros_like(W.T), atol=1e-8), "pInvGramSchmidt weights do not satisfy normal equations with atol=1e-8"

def GramSchmidt_test_QR_square():
    ns = np.arange(2,10)
    for n in ns:
        X = np.random.randn(n,n)
        Q, R = alc.QR_con_GS(X)
        assert is_QR_decomposition(X, Q, R, tol=1e-5), f"GramSchmidt_test for square matrix with size {n} x {n} and tol=1e-5 failed"

def GramSchmidt_test_QR_rectangular():
    ns = np.arange(2,9)
    ps = np.arange(8, 16)
    sizes = np.array(list(zip(np.repeat(ns,len(ns)), np.tile(ps, len(ps)))))
    for n, p in sizes:
        X = np.random.randn(p,n)
        Q, R = alc.QR_con_GS(X)

        assert is_QR_decomposition(X, Q, R, tol=1e-5), f"GramSchmidt_test for square matrix with size {n} x {n} and tol=1e-5 failed"


def get_arg_names(func):
    sig = inspect.signature(func)
    return [p.name for p in sig.parameters.values()]
 

print("------------------------------------------------------------")
GramSchmidt_test_QR_square()
print("GramSchmidt square matrixes test passed")
GramSchmidt_test_QR_rectangular()
print("GramSchmidt tall rectangular matrixes test passed")
pInvGramSchmidt_test_identity()
print("pInvGramSchmidt identity tests passed.")
pInvGramSchmidt_test_identity_wide_zero_error()
print("pInvGramSchmidt tall identity tests passed.")
pInvGramSchmidt_test_identity_wide_known_error()
print("pInvGramSchmidt tall identity with known error tests passed.")
pInvGramSchmidt_test_random_wide_known_error()
print("pInvGramSchmidt tall random with known error tests passed.")
pInvGramSchmidt_test_solves_normal_eq()
print("pInvGramSchmidt_test_solves_normal_eq tests passed.")