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
    return celdasACeroSi(matriz,lambda x,y:y!=x)
     

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
    assert x.shape[0]==m and x.shape[1]==1 , "el vector x tiene que ser del tamaño de #columnas_de_A x 1"
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

main()
