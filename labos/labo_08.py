import numpy
from modulo_alc import traspuesta,productoMatricial,diagRH
def svd_reducida(A,k="max",tol=1e-15):
    A_t = traspuesta(A) 
    A_X_At = productoMatricial(A,A_t)
    S,D = diagRH(A_X_At)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):

    return None 