import numpy as np
import labo_0
Matriz = np.ndarray 


def calculaLU(A:Matriz):
    resultado = elim_gaussiana(A)
    if resultado is None:
        return None,None,0
    return resultado

def res_tri(L,B,inferior=True):
    if L is None: return None
    rango_filas,rango_columnas = L.shape
    X = np.zeros(rango_filas)
    if inferior:
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
        if U[pivote][pivote] == 0:
            return 
        for fila_j in range(pivote+1,filas):
            f = U[fila_j][pivote] / U[pivote][pivote]
            L[fila_j][pivote] = f
            for columna_k in range(columnas):
                cant_op = cant_op+1
                U[fila_j][columna_k] = U[fila_j][columna_k] - f*U[pivote][columna_k] 
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
    return L, U, cant_op

def determinante_matriz_escalonada(A:Matriz):
    res =1 
    for x in range(A.shape[0]):
        res *=A[x][x]
    return res
def inversa(A:Matriz):
    L,U,_ = calculaLU(A)
    if determinante_matriz_escalonada(A) ==0 or L is None or U is None: return None
    rango_fila,rango_columna = A.shape
    inversa:list = [] 
    matriz_identidad = np.identity(rango_fila)
    for columna in range(rango_columna):
        y = res_tri(L,matriz_identidad[:,columna],True)
        inversa.append(res_tri(U,y,False))
        continue
    return np.column_stack(inversa)
def calculaLDV(A:Matriz):
    L,U,nops = calculaLU(A)
    if L is None or U is None: return None,None,None,0
    D = labo_0.diagonal(U)
    V = U
    for x in range(D.shape[0]):
        celda_diagonal = V[x][x]
        for y in range(D.shape[1]):
            nops +=1
            if x==y and V[x][x] != 0:
                    V[x][x] = 1.0
            elif x!=y and V[x][x]!=0:
                    V[x][y] = V[x][y]/celda_diagonal   
    return L,D,V,nops

def es_simetrica(A:Matriz,atol=1e-8):
    if not labo_0.esCuadrada(A): return False
    traspuesta_A = labo_0.traspuesta(A)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            if np.abs(A[x][y] - traspuesta_A[x][y]) > atol:
                return False
    return True  
def     esSDP(A:Matriz,atol=1e-10):
    if not es_simetrica(A,atol): return False
    _,D,_,_ = calculaLDV(A)
    if D is None: return False
    for x in range(D.shape[0]):
        if D[x][x] <= atol:
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

# Tests L04-LU

# Tests LU

L0 = np.array([[1,0,0],[0,1,0],[1,1,1]])
U0 = np.array([[10,1,0],[0,2,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(np.allclose(L,L0))
assert(np.allclose(U,U0))


L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(not np.allclose(L,L0))
assert(not np.allclose(U,U0))
assert(np.allclose(L,L0,atol=1e-3))
assert(np.allclose(U,U0,atol=1e-3))
#assert(nops == 13)

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,0,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(L is None)
assert(U is None)
assert(nops == 0)

## Tests res_tri

A = np.array([[1,0,0],[1,1,0],[1,1,1]])
b = np.array([1,1,1])
assert(np.allclose(res_tri(A,b),np.array([1,0,0])))
b = np.array([0,1,0])
assert(np.allclose(res_tri(A,b),np.array([0,1,-1])))
b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b),np.array([-1,2,-2])))
b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([-1,1,-1])))

A = np.array([[3,2,1],[0,2,1],[0,0,1]])
b = np.array([3,2,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1/3,1/2,1])))

A = np.array([[1,-1,1],[0,1,-1],[0,0,1]])
b = np.array([1,0,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1,1,1])))

# Test inversa

ntest = 10
iter = 0
while iter < ntest:
    A = np.random.random((4,4))
    A_ = inversa(A)
    if not A_ is None:
        assert(np.allclose(np.linalg.inv(A),A_))
        iter += 1

# Matriz singular devería devolver None
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
assert(inversa(A) is None)




# Test LDV:

L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
D0 = np.diag([1,2,3])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ D0  @ V0
L,D,V,_ = calculaLDV(A)
assert(np.allclose(L,L0))
assert(np.allclose(D,D0))
assert(np.allclose(V,V0))

L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
D0 = np.diag([3,2,1])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
A =  L0 @ D0  @ V0
L,D,V,nops = calculaLDV(A)
assert(np.allclose(L,L0,1e-3))
assert(np.allclose(D,D0,1e-3))
assert(np.allclose(V,V0,1e-3))

# Tests SDP

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
A = L0 @ D0 @ L0.T
assert(esSDP(A))

D0 = np.diag([1,-1,1])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

D0 = np.diag([1,1,1e-16])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
V0 = np.array([[1,0,0],[1,1,0],[1,1+1e-10,1]]).T
A = L0 @ D0 @ V0
assert(not esSDP(A))