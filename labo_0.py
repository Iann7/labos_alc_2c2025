import numpy as np 
import numbers
Matriz = np.ndarray 
def main():
    matriz_2x2:Matriz = np.asarray([[1,2],[3,4]])
    matriz_3x3:Matriz = np.asarray([[1,2,3],[1,2,3],[1,2,3]])
    matriz_3x3_dominante:Matriz = np.asarray([[100,2,3],[1,200,3],[1,2,300]])
    matriz_1x2:Matriz = np.asarray([[1,2]])
    print(numeroAureo(2))

def esCuadrada(matriz:Matriz):
    return matriz.shape[0] == matriz.shape[1]



def celdasACeroSi(matriz:Matriz,predicado):
    #TODO: despues del labo quizas conviene hacer una copia? 
    filas,columnas = matriz.shape
    for x in range(filas):
        for y in range(columnas):
            if predicado(x,y):
                matriz[x][y] = 0
    return matriz

def trianguloSuperior(matriz:Matriz):
    return  celdasACeroSi(matriz,lambda x,y: y<=x)
    


def trianguloInferior(matriz:Matriz):
    return celdasACeroSi(matriz,lambda x,y:y>=x)
     

def diagonal(matriz:Matriz):
    return celdasACeroSi(matriz.copy(),lambda x,y:y!=x)
     

def traza(matriz:Matriz):
    assert not esCuadrada(matriz),"NO TIENE SENTIDO APLICAR TRAZA A UNA MATRIZ NO CUADRADA"
    suma = 0
    for x in range(len(matriz[0])):
        suma += matriz[x][x]
    return suma

def traspuesta(matriz:Matriz):
    fila,columna = matriz.shape
    array_transpuesto = [[0] * fila for _ in range(columna)]
    for x in range(fila):
        for y in range(columna):
            array_transpuesto[y][x] = matriz[x][y] 
    return np.asarray(array_transpuesto)

def calcularAx(A:Matriz,x:Matriz):
    n,m = A.shape
    resultado_ax = [0] * n
    assert x.shape==(m,1) or x.shape == (m,) , "el vector x tiene que ser del tamaño de #columnas_de_A x 1"
    for i in range(n):
        for j in range(m):
            resultado_ax[i] += A[i][j]*x[j]
    return np.asarray(resultado_ax)

def intercambiarFilas(A:Matriz,i,j):
    A[i,j] = A[j,i]
    return A

def sumarFilaMultiplo(A:Matriz,i,j,s):
    A[i] += s*A[j]
    return A

def esDiagonalmenteDominante(A:Matriz)->bool:
    assert esCuadrada(A) , "LA MATRIZ NO ES CUADRADA"
    fila,columna = A.shape
    for x in range(fila):
        suma_fila =0
        for y in range(columna):
            if y==x:continue
            suma_fila += A[x][y]
        if suma_fila > A[x][x]: return False
    return True 

def matrizCirculante(v):
    array_v = [[0] * len(v) for _ in range(len(v))]
    for x in range(len(v)):
        for y in range(len(v)):
            array_v[x][y] = v[(y-x)%len(v)]
    return np.asarray(array_v)

def matrizVandermonde(v):
    array_v = [[0] * len(v) for _ in range(len(v))]
    for x in range(len(v)):
        for y in range(len(v)):
            array_v[x][y] = v[x]**y
    return np.asarray(array_v)

def numeroAureo(n):
    f_k_y_f_kminus = np.asarray([[1],[0]])
    mascara_para_calcular = np.asarray([[1,1],[1,0]])
    for i in range(n):
        f_k_y_f_kminus = calcularAx(mascara_para_calcular,f_k_y_f_kminus)
    return f_k_y_f_kminus[1][0]

def matrizFibonacci(n):
    array_v = [[0] * n for _ in range(len(n))]
    array_v[0][1] = 1
    #TODO

def calcular_polinomio():
    #TODO
    return

def matrizHilbert(n):
    array_v = [[0] * n for _ in range(len(n))]
    for x in range(n):
        for y in range(n):
            array_v[x][y] = 1/(x+y+1)
    return np.assarray(array_v)

def row_echelon(M:Matriz):
    """ 
        Retorna la Matriz Escalonada por Filas 
    """
    A:Matriz = np.copy(M)
    if (issubclass(A.dtype.type, np.integer)):
        A = A.astype(float)
    # Si A no tiene filas o columnas, ya esta escalonada
    f, c = A.shape
    if f == 0 or c == 0:
        return A
    # buscamos primer elemento no nulo de la primera columna
    # NOTE: El univo Cambio Que hicimos aca fue en vez de agarrar 0 conseguimos la fila cuyo elemento en la columna 0 tuviese mayor modulo
    i = np.argmax(np.abs(A[:,0]))
    while i < f and A[i,0] == 0:
        i += 1
    if i == f:
        # si todos los elementos de la primera columna son ceros
        # escalonamos filas desde la segunda columna
        B = row_echelon(A[:,1:])
        # y volvemos a agregar la primera columna de zeros
        return np.block([A[:,:1], B])
    # intercambiamos filas i <-> 0, pues el primer cero aparece en la fila i
    if i > 0:
        A[[0,i],:] = A[[i,0],:]
    # PASO DE TRIANGULACION GAUSSIANA:
    # a las filas subsiguientes les restamos un multiplo de la primera
    A[1:,:] -= (A[0,:] / A[0,0]) * A[1:,0:1]
    # escalonamos desde la segunda fila y segunda columna en adelante
    B = row_echelon(A[1:,1:])
    # reconstruimos la matriz por bloques adosando a B la primera fila 
    # y la primera columna (de ceros)
    return np.block([ [A[:1,:]], [ A[1:,:1], B] ])