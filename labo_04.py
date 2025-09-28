import numpy as np
import labo_0
Matriz = np.ndarray 


def calculaLU(A:Matriz):
    n,m = A.shape
    L,U,_ = elim_gaussiana(A)
    return L,U

def res_tri(L,B,triangular_inferior):
    rango_filas,rango_columnas = L.shape
    X = np.zeros(rango_filas)
    if triangular_inferior:
        resolver_fila(L, B, rango_filas, rango_columnas, X,lambda n:range(n),
                      lambda fila_actual,m:range(0,fila_actual))
    else:  
        resolver_fila(L, B, rango_filas, rango_columnas, X,
                      lambda n:range(n-1,-1,-1),
                      lambda fila_actual,m:range(fila_actual+1,m))
    return X

def resolver_fila(L, B, rango_filas, rango_columnas, X,condicion_for_fila,condicion_for_columna):
    for fila_actual in condicion_for_fila(rango_filas):
        res=B[fila_actual]
        for columna_actual in condicion_for_columna(fila_actual,rango_columnas):
            res = (res-L[fila_actual][columna_actual]*X[columna_actual])
        X[fila_actual] = res/L[fila_actual][fila_actual]


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

def inversa(A:Matriz):
    L,U = calculaLU(A)
    rango_fila,rango_columna = A.shape
    inversa:list = [] 
    matriz_identidad = np.identity(rango_fila)
    for columna in range(rango_columna):
        y = res_tri(L,matriz_identidad[:,columna],True)
        inversa.append(res_tri(U,y,False))
        continue
    return np.column_stack(inversa)
def calculaLDV(A:Matriz):
    L,U = calculaLU(A)
    D = labo_0.diagonal(U)
    V = U
    for x in range(D.shape[0]):
        if V[x][x] != 0:
            V[x][x] = V[x][x] / V[x][x] 
    return L,D,V

def es_simetrica(A:Matriz,atol=1e-8):
    if not labo_0.esCuadrada(A): return False
    traspuesta_A = labo_0.traspuesta(A)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            if np.abs(A[x][y] - traspuesta_A[x][y]) > atol:
                return False
    return True  
def esSDP(A:Matriz,atol=1e-8):
    if not es_simetrica(A,atol): return False
    _,D,_ = calculaLDV(A)
    for x in range(D.shape[0]):
        if D[x][x] <= 0:
            return False
    return True


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



print(calculaLDV(np.array([[1,2,3],[4,5,6],[7,8,9]])))