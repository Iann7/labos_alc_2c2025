import labo_0
import labo_01
import numpy as np
Matriz = np.ndarray 

#similar a reduce en python
def reducir_matriz(x:Matriz,p,operador):
    n = x.shape[0]
    sumatoria_celdas = 0
    for i in range(n):
        sumatoria_celdas = operador(sumatoria_celdas,x[i],p)
    return sumatoria_celdas

def norma(x:Matriz,p):
    sumatoria_celdas:float = 0
    if p!= 'inf':
        sumatoria_celdas = reducir_matriz(x,p,lambda sum,celda,p: sum + np.abs(celda)**p)
        sumatoria_celdas = sumatoria_celdas**(1/p)
    else:
        sumatoria_celdas = reducir_matriz(x,p,lambda sum,celda,p: max(sum,np.abs(celda)))
    return sumatoria_celdas

def normaliza(X:list[Matriz],p):
    res:list[Matriz] = []
    for x in X:
        res.append((1/norma(x,p)*x))
    return res


def suma_maxima_sobre_eje(A:Matriz,OUTER,INNER,acceso_a_celda):
    mejor_suma = -np.inf
    for x in range(OUTER):
        suma = 0 
        for y in range(INNER):
            suma = suma + np.abs(acceso_a_celda(A,x,y))
        if suma > mejor_suma:
            mejor_suma = suma
    return mejor_suma
def normaExacta(A:Matriz,p=[1,'inf']):
    n,m = A.shape
    if p!='inf' and p!=1: return None
    if p=='inf':
        return suma_maxima_sobre_eje(A,n,m,lambda A,x,y:A[x][y])
    else: 
        return suma_maxima_sobre_eje(A,m,n,lambda A,x,y:A[y][x])

def normaMatMC(A:Matriz,q,p,Np):
    n,m  = A.shape
    mejor_valor = -np.inf
    mejor_x = None 
    lista_vectores_aleatorios = []
    for _ in range(Np):
        lista_vectores_aleatorios.append(np.random.randn(m))
    lista_vectores_aleatorios = normaliza(lista_vectores_aleatorios,p)
    for i in range(Np):
        A_x = labo_0.calcularAx(A,lista_vectores_aleatorios[i])
        norma_A_x = norma(A_x,q)
        if norma_A_x > mejor_valor:
            mejor_valor = norma_A_x
            mejor_x = lista_vectores_aleatorios[i]
    return mejor_valor,mejor_x

def condMC(A,p,Np):
    norma_A = normaMatMC(A,p,p,Np)[0]
    norma_A_inversa = normaMatMC(np.linalg.inv(A),p,p,Np)[0]
    return norma_A*norma_A_inversa
def condExacta(A,p):
    norma_A = normaExacta(A,p)
    norma_A_inversa = normaExacta(np.linalg.inv(A),p)
    return norma_A*norma_A_inversa
# Tests L03-Normas

# Tests norma
assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

# Tests normaliza
for x in normaliza([np.array([1]*k) for k in range(1,11)],2):
    assert(np.allclose(norma(x,2),1))
for x in normaliza([np.array([1]*k) for k in range(2,11)],1):
    assert(not np.allclose(norma(x,2),1) )
for x in normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
    assert( np.allclose(norma(x,'inf'),1) )


## Tests normaExacta

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)),1)<=10)
assert(normaExacta(np.random.random((4,4)),'inf')<=4)

## Test normaMC
#
#nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
#assert(np.allclose(nMC[0],1,atol=1e-3))
#assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
#assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))
#
#nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
#assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
#assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
#
#A = np.array([[1,2],[3,4]])
#nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
#assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

# Test condMC

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

# Test condExacta

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,1)
normaA_ = normaExacta(A_,1)
condA = condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,'inf')
normaA_ = normaExacta(A_,'inf')
condA = condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA))