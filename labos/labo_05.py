import numpy as np
from labo_0 import calcularAx,traspuesta
import labo_0
import labo_03
Matriz = np.ndarray

def QR_con_GS(A:Matriz,tol=1e-12,retorna_nops=False):
    rango_filas,rango_columnas = A.shape
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    Q[0] = labo_03.normaliza([A[0]],p=2)[0]
    for j in range(rango_columnas):
        q_temp = A[:,j].copy()
        for k in range(j):
            R[k][j] = labo_0.calcularAx((np.asarray(Q[:,k])),q_temp)
            q_temp = q_temp - R[k][j]*Q[:,k]
        R[j][j] = labo_03.norma(q_temp,2)
        Q[:,j] = q_temp/R[j][j]
    return Q,R

def QR_con_HH(A:Matriz,tol=1e-12):
    R = A.copy()
    m,n = R.shape
    Q = np.identity(m)
    for k in range(n):
        I = np.identity(m)
        x = R[k:m,k]
        e = I[k:m,k]
        alpha = -np.sign(x[0]) * labo_03.norma(x,2)
        u = x-alpha*e
        norma_u = labo_03.norma(u,2)
        v = u/norma_u
        if norma_u > tol:
            v = v.reshape(-1, 1)  # Convertir a vector columna
            R[k:m, k:n] = R[k:m, k:n] - 2 * v @ (v.T @ R[k:m, k:n])
            Q[:, k:m] = Q[:, k:m] - 2 * (Q[:, k:m] @ v) @ v.T
    return Q,R

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)