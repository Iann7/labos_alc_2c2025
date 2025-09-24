import numpy as np
import labo_0
Matriz = np.ndarray 


def calculaLU(A:Matriz):
    n,m = A.shape
    L,U,_ = elim_gaussiana(A)
    return L,U

def res_tri(L,B,triangular_inferior):
    filas,columnas = L.shape
    X = np.zeros(L.shape)
    if triangular_inferior:
        return
    else:
        res = B[x]
        for x in range(filas,0):
            for y in range(x+1,columnas):
                res = res - L[x][y]
            X[x][y]=res
    return 


def elim_gaussiana(A:Matriz):
    cant_op = 0
    filas=A.shape[0]
    columnas=A.shape[1]
    Ac = A.copy()
    L = np.eye(filas)
    U = A.copy()
    if filas!=columnas:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    for pivote in range(columnas):
        assert A[pivote][pivote] != 0, "ERROR,UNO DE LOS ELEMENTOS EN LAS DIAGONALES ES 0"
        for fila_j in range(pivote+1,filas):
            f = U[fila_j][pivote] / U[pivote][pivote]
            L[fila_j][pivote] = f
            for columna_k in range(columnas):
                cant_op = cant_op+1
                U[fila_j][columna_k] = U[fila_j][columna_k] - f*U[pivote][columna_k] 
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
    return L, U, cant_op


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
