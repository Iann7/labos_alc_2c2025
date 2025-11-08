import numpy as np 
import numbers
import labo_0
import matplotlib.pyplot as pyplot 
Matriz = np.ndarray
Double = np.longdouble 
def main():
    matriz_2x2:Matriz = np.asarray([[1,2],[3,4]])
    matriz_3x3:Matriz = np.asarray([[1,2,3],[1,2,3],[1,2,3]])
    matriz_3x3_dominante:Matriz = np.asarray([[100,2,3],[1,200,3],[1,2,300]])
    matriz_1x2:Matriz = np.asarray([[1,2]])
    cerotres : Double = 0.3
    cero25 : Double = 0.25
    res: Double = cerotres+cero25
    print(np.sqrt(2)**2 + 200000 - 200002)
    res = cerotres-cero25
    x = np.linspace(0,5*1e-8,100)
    pyplot.plot(x,func_a(x), label="a")
    pyplot.plot(x,func_b(x), label="b")
    pyplot.legend()
    pyplot.savefig("./test.png")

def func_a(x):
    return np.sqrt(2*x**2+1)-1

def func_b(x):
    return 2*x**2/(np.sqrt(2*x**2+1)+1)
    
def error(x,y):
    return np.abs(x-y)
    #Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64

def error_relativo(x,y):
    return np.abs(error(x,y)/np.abs(x))
    #Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64
    
def matricesIguales(A:Matriz,B:Matriz):
    #Devuelve True si ambas matrices son iguales y False en otro caso.
    #Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    if A.shape != B.shape: return False
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            a= A[x][y]
            b= B[x][y]
            if error(A[x][y],B[x][y])>= 1e-8:
                return False
    return True  

import numpy as np

def sonIguales(x, y, atol=1e-8):
    return np.allclose(error(x, y), 0, atol=atol)

# Tests para sonIguales
assert not sonIguales(1, 1.1)
assert sonIguales(1, 1 + np.finfo('float64').eps)
assert not sonIguales(1, 1 + np.finfo('float32').eps)
assert not sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps)
assert sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3)

# Tests para error relativo
assert np.allclose(error_relativo(1, 1.1), 0.1)
assert np.allclose(error_relativo(2, 1), 0.5)
assert np.allclose(error_relativo(-1, -1), 0)
assert np.allclose(error_relativo(1, -1), 2)

# Tests para matricesIguales
assert matricesIguales(np.diag([1, 1]), np.eye(2))
assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
assert not matricesIguales(np.array([[1, 2], [3, 4]]).T, np.array([[1, 2], [3, 4]]))

