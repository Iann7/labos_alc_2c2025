import matplotlib.pyplot as plt
import numpy as np
from alc import *


def armarMatrizDeConfusion(W, X_v, Y_v):
    def actualizarMatrizDeConfusion(matrizConfusion, esPerro, esPerro_Predicho):
        if esPerro and esPerro_Predicho:
            matrizConfusion[1][1] += 1
        elif esPerro and not esPerro_Predicho:
            matrizConfusion[1][0] += 1
        elif not esPerro and esPerro_Predicho:     
            matrizConfusion[0][1] += 1
        else:     
            matrizConfusion[0][0] += 1


    Y_Predicho = productoMatricial(W, X_v)
    matrizConfusion = np.zeros((2,2))
    row_Y, col_Y  = Y_Predicho.shape # Tienen todas el mismo tamaño
    for i in range(col_Y):
        esPerro = Y_v[0][i] == 1
        esPerro_Predicho = Y_Predicho[0][i] >= Y_Predicho[1][i]
        actualizarMatrizDeConfusion(matrizConfusion, esPerro, esPerro_Predicho)
            
    return matrizConfusion

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Gato", "Perro"])
    ax.set_yticklabels(["Gato", "Perro"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def calcularYPlotearMatrizDeConfusion(X_t, Y_t, X_v, Y_v, metodo="Cholesky"):
    if metodo == "Cholesky":
        W = fullyConnectedLineal_Cholesky(X_t, Y_t)
    elif metodo == "QR":
        W = fullyConnectedLineal_QR(X_t, Y_t)
    elif metodo == "SVD":
        W = fullyConnectedLineal_SVD(X_t, Y_t)
    else:
        raise ValueError("Método no reconocido. Usar 'Cholesky', 'QR' o 'SVD'.")
    
    print(f"FINALIZO {metodo}")
    matrizConfusion = armarMatrizDeConfusion(W, X_v, Y_v)
    plot_confusion_matrix(matrizConfusion, f"Matriz de Confusión - {metodo}")
    return matrizConfusion