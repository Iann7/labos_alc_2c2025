from matplotlib import pyplot as plt
import numpy as np
import inspect

from entrega import alc as main



def is_QR_decomposition(A, Q, R, tol=1e-5):
    if not np.allclose(A, Q @ R, atol=tol):
        return False
    if not np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol):
        return False
    if not np.allclose(R, np.triu(R), atol=tol):
        return False
    return True

def householder_test_row_matrix():
    for _ in range(500):
        A = np.random.randn(1, 10)
        Q, R = main.QR_con_HH(A)
        assert Q.shape == (1, 1), "Q shape incorrect for row matrix"
        assert R.shape == (1, 10), "R shape incorrect for row matrix"
        assert is_QR_decomposition(A, Q, R), "Householder QR failed for row matrix"

def householder_test_column_matrix():
    for _ in range(500):
        A = np.random.randn(10, 1)
        Q, R = main.QR_con_HH(A)
        # assert Q.shape == (10, 1), "Q shape incorrect for column matrix"
        # assert R.shape == (1, 1), "R shape incorrect for column matrix"
        assert is_QR_decomposition(A, Q, R), "Householder QR failed for column matrix"

def householder_test_tall_matrix():
    for k in range(500):
        A = np.random.randn(7,5)
        Q, R = main.QR_con_HH(A)
        # assert Q.shape == (7,5), "Q shape incorrect for tall matrix"
        # assert R.shape == (5,5), "R shape incorrect for tall matrix"
        assert is_QR_decomposition(A, Q, R), "Householder QR failed for tall matrix"

def householder_test_wide_matrix():
    for k in range(500):
        A = np.random.randn(5,7)
        Q, R = main.QR_con_HH(A)
        assert Q.shape == (5,5), "Q shape incorrect for wide matrix"
        assert R.shape == (5,7), "R shape incorrect for wide matrix"
        assert is_QR_decomposition(A, Q, R), "Householder QR failed for wide matrix"

def householder_sign_handling_test():
    for i in range(500):
        A = np.random.randn(1, 10)
        Q, R = main.QR_con_HH(A)
        assert Q.shape == (1, 1), "Q shape incorrect for row matrix"
        assert R.shape == (1, 10), "R shape incorrect for row matrix"
        expected_R = A
        expected_Q = np.array([[1]])
        assert np.allclose(R[0], expected_R), f"Case {i}: Not handling sign as we expected for Householder QR on row matrix"
        assert np.allclose(Q, expected_Q),  f"Case {i}: Not handling sign as we expected for Householder QR on row matrix"

        A = -A
        Q, R = main.QR_con_HH(A)
        assert Q.shape == (1, 1), "Q shape incorrect for row matrix"
        assert R.shape == (1, 10), "R shape incorrect for row matrix"
        expected_R = A
        assert np.allclose(R[0], expected_R), f"Case {i}: Not handling sign as we expected for Householder QR on row matrix"
        assert np.allclose(Q, expected_Q),  f"Case {i}: Not handling sign as we expected for Householder QR on row matrix"

def householder_test_its_householder_easy_case():
    for i in range(100):
        A = np.random.randn(2, 2)
        Q, R = main.QR_con_HH(A)
        assert Q.shape == (2, 2), f"Case {i}: Q shape incorrect for easy 2x2 case"
        assert R.shape == (2, 2), f"Case {i}: R shape incorrect for easy 2x2 case"
        assert is_QR_decomposition(A, Q, R), f"Case {i}: Householder QR failed for easy 2x2 case"

def householder_test_two_copied_matrix():
    for k in range(500):
        A = np.random.randn(8,8)
        A = np.hstack((A, A.copy()))
        Q, R = main.QR_con_HH(A)
        assert Q.shape == (8,8), "Q shape incorrect for two-copied matrix"
        assert R.shape == (8,16), "R shape incorrect for two-copied matrix"
        assert is_QR_decomposition(A, Q, R), "Householder QR failed for two-copied matrix"
        # como las ultimas 8 columnas de A son copia de las primeras 8, las ultimas 8 columnas de R deberian ser iguales a las primeras 8
        assert np.allclose(R[:,8:], R[:,:8]), "R columns do not match for two-copied matrix"

def householder_test_permutation_copy_shuffle():
    for k in range(500):
        A = np.random.randn(6,6)
        A = np.hstack((A, A.copy()))
        indices = [x%6 for x in range(12)]
        # tenemos que mezclar los indices pero recordar cual matchea con cual
        np.random.shuffle(indices)
        A = A[:, indices]
        Q, R = main.QR_con_HH(A)
        assert Q.shape == (6,6), "Q shape incorrect for permutation copy shuffle matrix"
        assert R.shape == (6,12), "R shape incorrect for permutation copy shuffle matrix"
        assert is_QR_decomposition(A, Q, R), "Householder QR failed for permutation copy shuffle matrix"
        for i, column in enumerate(R.T):
            original_index = indices[i]
            # buscar el índice correspondiente en R
            j = 0
            while j < len(indices):
                if indices[j] == original_index and j != i:
                    break
                j += 1
            expected_column = R[:, j]
            assert np.allclose(column, expected_column), "R columns do not match after permutation shuffle"

def housholder_easy_diag_matrix_test():
    A = np.eye(3)
    Q, R = main.QR_con_HH(A)
    assert is_QR_decomposition(A,Q,R), "Householder QR failed on identity matrix"

    A[2,2] = -1
    Q, R = main.QR_con_HH(A)
    assert is_QR_decomposition(A,Q,R), "Householder QR failed on modified identity matrix"

    A[1,1] = 0
    Q, R = main.QR_con_HH(A)
    assert is_QR_decomposition(A,Q,R), "Householder QR failed on modified identity matrix with zero"

def householder_test_scalar_matrix():
    A = np.array([[5]])
    Q, R = main.QR_con_HH(A)
    assert is_QR_decomposition(A,Q,R), "Householder QR failed on scalar matrix"

    A = np.array([[-3]])
    Q, R = main.QR_con_HH(A)
    assert is_QR_decomposition(A,Q,R), "Householder QR failed on negative scalar matrix"

def householder_test_zero_matrix():
    A = np.zeros((4,4))
    Q, R = main.QR_con_HH(A)
    assert is_QR_decomposition(A,Q,R), "Householder QR failed on zero matrix"

def householder_test_zero_with_column():
    A = np.zeros((4,4))
    A[:,2] = np.array([1,2,3,4])
    Q, R = main.QR_con_HH(A)
    # Nota, tecnicamente si no atajan el tema de elementos nulos en R les puede quedar una matriz no columan para R, pero esta ok igual asi que no lo chequeamos
    assert is_QR_decomposition(A, Q, R), "Householder QR failed on zero matrix with one column"


def pinvHouseHolder_wrapper(X, Y):
    fun_params = get_arg_names(main.pinvHouseHolder)
    params = {"X": X, "Y": Y}
    if "Q" in fun_params and "R" in fun_params:
        #Q, R = np.linalg.qr(X.T, mode= 'full')  # Usamos la de numpy para no depender de que el QR con HH ande bien
        Q, R = main.QR_con_HH(X.T) 
        
        # Descomentar si usan la version completa
        # n = X.T.shape[1]
        # Q = Q[:, :n]            
        # R = R[:n, :n] 
        params["Q"] = Q
        params["R"] = R
        params.pop("X")  # no hace falta mandar X si mandamos Q y R
    return main.pinvHouseHolder(**params)

def pinvHouseHolder_test_identity():
    X = np.eye(3)
    for _ in range(100):
        Y = np.random.randn(3, 3)
        W = pinvHouseHolder_wrapper(X, Y)
        Y_pred = W @ X
        assert Y_pred.shape == Y.shape, f"Y_pred shape incorrect for identity matrix. Expected {Y.shape}, got {Y_pred.shape}"
        assert np.allclose(Y_pred, Y), "pinvHouseHolder failed on identity matrix"

def pinvHouseHolder_test_identity_wide_zero_error():
    X = np.eye(5,7)
    for _ in range(500):
        W_true = np.random.randn(32, 5)
        Y = W_true @ X  # esta en la imagen de X, asi que el error deberia poder ser 0
        W = pinvHouseHolder_wrapper(X, Y)
        Y_pred = W @ X
        assert Y_pred.shape == Y.shape, f"Y_pred shape incorrect for tall identity matrix. Expected {Y.shape}, got {Y_pred.shape}"
        assert np.allclose(Y_pred, Y), "pinvHouseHolder failed on tall identity matrix"
        assert np.allclose(W, W_true), "pinvHouseHolder did not recover the correct weights for tall identity matrix"


def pinvHouseHolder_test_identity_wide_known_error():
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
        W = pinvHouseHolder_wrapper(X, Y)
        Y_pred = W @ X
        error = Y - Y_pred
        error = np.linalg.norm(error, ord='fro')
        error_expected = np.linalg.norm(E, ord='fro')
        assert Y_pred.shape == Y.shape, f"Y_pred shape incorrect for tall identity matrix with known error. Expected {Y.shape}, got {Y_pred.shape}"
        assert np.allclose(error, error_expected, atol=1e-5), f"pinvHouseHolder did not achieve the expected error {error_expected} for tall identity matrix with known error, got {error}"
        assert np.allclose(W, W_true, atol=1e-5), "pinvHouseHolder did not recover the correct weights for tall identity matrix with known error"


def pinvHouseHolder_test_random_wide_known_error():
    for _ in range(500):
        X = np.random.randn(5,7) * 100 # para numeros mas grandes 
        # no tan facil como antes pero usando la SVD de numpy se puede. Dado que X es aleatoria, el rango deberia ser completo
        U, _, _ = np.linalg.svd(X.T, full_matrices=True)
        Q_null = U[:, 5:]  # base del espacio nulo de XT

        E = 100 * np.random.randn(32, Q_null.shape[1]) @ Q_null.T  # ruido en el espacio ortogonal
        W_true = np.random.randn(32, 5)
        Y = W_true @ X + E

        W = pinvHouseHolder_wrapper(X, Y)

        Y_pred = W @ X
        error = Y - Y_pred
        error = np.linalg.norm(error, ord='fro')
        error_expected = np.linalg.norm(E, ord='fro')
        assert Y_pred.shape == Y.shape, f"Y_pred shape incorrect for tall random matrix with known error. Expected {Y.shape}, got {Y_pred.shape}"
        assert np.allclose(error, error_expected, atol=1e-5), f"pinvHouseHolder did not achieve the expected error {error_expected} for tall random matrix with known error, got {error}"
        assert np.allclose(W, W_true, atol=1e-5), "pinvHouseHolder did not recover the correct weights for tall random matrix with known error"


def pinvHouseHolder_test_using_QR_con_HH():
    # Este es el unico que solo aplica a householder, usa internamente el tema del signo. Si falla puede estar bien igual
    for _ in range(500):
        Y = np.random.randn(32,7)
        X = np.eye(5,7)
        X[0,0] = -1  # para forzar el tema del signo
        Q, R = main.QR_con_HH(X.T) # Asume que Q y R estan reducidos (Q de mxn, R de nxn)
        # Descomentar si estan usando la version completa
        n = X.T.shape[1]
        Q = Q[:, :n]            
        V = Q.copy()
        V[:,0] = -V[:,0] 
        W_true = Y @ V
        W = pinvHouseHolder_wrapper(X, Y)
        # Verificacion, la primer columna puede estar de signo opuesto. 
        assert np.allclose(W, W_true, atol=1e-5), "pinvHouseHolder did not recover the correct weights when using Householder QR internals" 

def get_arg_names(func):
    sig = inspect.signature(func)
    return [p.name for p in sig.parameters.values()]

if __name__ == "__main__":
    householder_test = False
    print("--------------------------------")
    print("Starting Householder QR tests...")
    print("--------------------------------")

    householder_test_zero_matrix() # caso borde
    print("Zero matrix Householder QR tests passed.")
    householder_test_zero_with_column() # caso borde
    print("Zero matrix with one column Householder QR tests passed.")
    housholder_easy_diag_matrix_test()
    print("Easy diagonal matrix Householder QR tests passed.")
    householder_test_scalar_matrix()
    print("Scalar matrix Householder QR tests passed.")
    householder_test_tall_matrix()
    print("Tall matrix Householder QR tests passed.")
    householder_test_column_matrix()
    print("Column matrix Householder QR tests passed.")
    householder_test_its_householder_easy_case()
    print("Easy 2x2 case Householder QR tests passed.")

    # TESTS ANCHOS

    print("--------------------------------")
    print("Starting pinvHouseHolder tests...")
    print("--------------------------------")

    pinvHouseHolder_test_identity()
    print("pinvHouseHolder identity tests passed.")
    pinvHouseHolder_test_identity_wide_zero_error()
    print("pinvHouseHolder tall identity tests passed.")
    pinvHouseHolder_test_identity_wide_known_error()
    print("pinvHouseHolder tall identity with known error tests passed.")

    pinvHouseHolder_test_random_wide_known_error()
    print("pinvHouseHolder tall random with known error tests passed.")

    pinvHouseHolder_test_using_QR_con_HH()
    print("pinvHouseHolder using Householder QR tests passed.")