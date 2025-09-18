import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import labo_0
import labo_01
Matriz = np.ndarray 
def pointsGrid(esquinas):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 46),
                        np.linspace(esquinas[0,1], esquinas[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 10),
                        np.linspace(esquinas[0,1], esquinas[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
                         
    return wz

def proyectarPts(T, wz):
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = None
    ############### Insert code here!! ######################3    

    ############### Insert code here!! ######################3
    return xy

          
def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             

    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')    
    
def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)


def main():
    return
    #print('Ejecutar el programa')
    ## generar el tipo de transformacion dando valores a la matriz T
    #T = pd.read_csv('T.csv', header=None).values
    #corners = np.array([[0,0],[100,100]])
    ## corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    #wz = pointsGrid(corners)
    #vistform(T, wz, 'Deformar coordenadas')
    
    
if __name__ == "__main__":
    main()


def generarArrayCuadradoNulo2D(tamaño):
    return [[0] *tamaño for _ in range(tamaño)]

def rota(theta)->Matriz:
    return np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    # Recibe un angulo theta y retorna 
    # una matriz 2x2 que rota un vector dad en 
    # un angulo theta

def escala(s:Matriz):
    n = len(s)
    s_escalada = generarArrayCuadradoNulo2D(n)
    for x in range(n):
            s_escalada[x][x] = s[x]
    return np.asarray(s_escalada) 
def rota_y_escala(theta,s:Matriz):
    matriz_rotacion = rota(theta)
    matriz_escalada = escala(s)
    columna_1:Matriz = labo_0.calcularAx(matriz_rotacion,matriz_escalada[0])
    columna_2:Matriz = labo_0.calcularAx(matriz_rotacion,matriz_escalada[1])
    matriz_mas_columnas_vacias = np.column_stack([columna_1,columna_2,np.asarray([[0],[0]])])
    return np.vstack([matriz_mas_columnas_vacias,np.array([0,0,1])])
def afin(theta,s,b):
    #Recibe un angulo theta una tira de numeros s en R2 y un vector b en R2
    #Retorna una matriz de 3x3 que rota el vector en un angulo theta 
    #luego lo escala en un factor s y por ultimo lo mueve en un valor fijo b
    rotada_y_escalada = rota_y_escala(theta,s)
    rotada_y_escalada[0][2] = b[0]
    rotada_y_escalada[1][2] = b[1]
    return rotada_y_escalada

def trans_afin(v,theta,s,b):
    return labo_0.calcularAx(afin(theta,s,b),v) 


#Test para rota 
assert np.allclose(rota(0), np.eye(2))
assert np.allclose(rota(np.pi / 2), np.array([[0, -1], [1, 0]]))
assert np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]]))

#Test para escala 
assert np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]]))
# Test 3: Escalar el vector [0.5, 0.25] debería dar la matriz diagonal [[0.5, 0], [0, 0.25]]
assert np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]]))

# Tests para rota y escala
assert np.allclose(rota_y_escala(0, [2, 3]),np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]]))

assert np.allclose(rota_y_escala(np.pi / 2, [1, 1]),np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))

assert np.allclose(rota_y_escala(np.pi, [2, 2]),np.array([[-2, 0, 0], [0, -2, 0], [0, 0, 1]]))
# Tests para la función afin
import numpy as np

assert np.allclose(
    afin(0, [1, 1], [1, 2]),
    np.array([
        [1, 0, 1],
        [0, 1, 2],
        [0, 0, 1]
    ])
)

assert np.allclose(
    afin(np.pi / 2, [1, 1], [0, 0]),
    np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
)

assert np.allclose(
    afin(0, [2, 3], [1, 1]),
    np.array([
        [2, 0, 1],
        [0, 3, 1],
        [0, 0, 1]
    ])
)
print("funciona todo")

