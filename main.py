import numpy as np
import modulo_alc
from modulo_alc import svd_reducida,productoMatricial,traspuesta
path_base = "./cats_and_dogs"

def main():
    X_t, Y_t, X_v, Y_v = cargarDataset(path_base)
    W_Cholesky = fullyConnectedLineal_Cholesky(X_t, Y_t)
    W_SVD = fullyConnectedLineal_SVD(X_t, Y_t)
    W_QR = fullyConnectedLineal_QR(X_t, Y_t)

    print(f"W_QR: {W_QR}")

    return 

def fullyConnectedLineal_Cholesky(X, Y):
    return None 



def list_to_diag(X:list):
    n = len(X)
    matriz_diagonal:np.ndarray = np.zeros((n,n))
    for i in range(n):
        matriz_diagonal[i][i]=X[i]
    return matriz_diagonal
    
def fullyConnectedLineal_SVD(X:np.ndarray, Y:np.ndarray):
    n,p = X.shape
    U_de_x,Sigma_de_x,V_de_x = svd_reducida(X,k=n)
    U_de_x_traspuesta = traspuesta(U_de_x)
    inversa_de_Sigma_x = list_to_diag(1.0 / Sigma_de_x)  
    X_inversa = productoMatricial(productoMatricial(V_de_x,inversa_de_Sigma_x),U_de_x_traspuesta)
    return productoMatricial(Y,X_inversa)

def fullyConnectedLineal_QR(X, Y):
    # print(X)
    X = X[:,:10]
    Y = Y[:,:10]
    Q, R = modulo_alc.calculaQR(modulo_alc.traspuesta(X))
    # print(Q)
    # print(R)
    V_rows = []
    Qt = modulo_alc.traspuesta(Q)
    cols_Qt = Qt.shape[1]
    for i in range(cols_Qt):
        V_rows.append(modulo_alc.res_tri(R, Qt[:,i], False))
        print(V_rows[-1].shape)
    V = np.array(V_rows)
    print(f"V.shape: {V.shape}")
    return modulo_alc.productoMatricial(Y, V)


def generarY(vector, n):
    # checkear que valores sean 1 o 0
    cantUno = 0
    for val in vector:
        if val == 1:
            cantUno += 1
        if(val != 1 and val != 0):
            # Return error
            return -1
            
    if cantUno != 1:
        # Retornar error
        return -1

    Y = np.zeros((vector.shape[0], n))
    for i in range(n):
        Y[:,i] = vector
    
    return Y

    

    # checkear norma vector = 1
        

def cargarDataset(carpeta):
    dogs_t = np.load(carpeta + "/train/dogs/efficientnet_b3_embeddings.npy")
    cats_t = np.load(carpeta + "/train/cats/efficientnet_b3_embeddings.npy")
    dogs_v = np.load(carpeta + "/val/dogs/efficientnet_b3_embeddings.npy")
    cats_v = np.load(carpeta + "/val/cats/efficientnet_b3_embeddings.npy")


    Y_t_dogs = generarY(np.array([1,0]), dogs_t.shape[1])
    Y_t_cats = generarY(np.array([0,1]), cats_t.shape[1])
    Y_v_dogs = generarY(np.array([1,0]), dogs_v.shape[1])
    Y_v_cats = generarY(np.array([0,1]), cats_v.shape[1])


    X_t = np.column_stack((dogs_t, cats_t)) 
    Y_t = np.column_stack((Y_t_dogs, Y_t_cats))
    X_v = np.column_stack((dogs_v, cats_v)) 
    Y_v = np.column_stack((Y_v_dogs, Y_v_cats))

    return X_t, Y_t, X_v, Y_v

main()
