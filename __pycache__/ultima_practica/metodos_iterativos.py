import numpy as np

def main():
    gauss_seidel()
    return

#REQ
# Square Matriz
# Linear
# Ideally Diagonally Dominant 
def gauss_seidel(A:np.ndarray,b:np.ndarray,atol:1e-18):
    cant_filas_A,cant_columnas_A = A.shape
    x = np.zeros((cant_columnas_A,1))
    for i in range(cant_filas_A):
        for j in range(cant_columnas_A):
            d = b[j]
            if (j!=i):
                d -= A[i][j] * x[i]
            x[i] = d/A[i][i] 
    return x

def jacobi(A:np.ndarray,b:np.ndarray,atol:1e-18):
    k=0
    while hasConverged():
        for i in range(A.shape[0]):
            d=0
            for j in range(A.shape[1]):
                if j!=i:
                    d= d+A[i][j]*x[j]
        x[i] = (b[i]-d)/A[i][i]
    return

def hasConverged()->bool:
    return False

main()