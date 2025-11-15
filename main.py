import numpy as np
import modulo_alc
from modulo_alc import svd_reducida,productoMatricial,traspuesta,matricesIguales,calculaCholesky,res_tri
path_base = "./cats_and_dogs"

def main():
    X_t, Y_t, X_v, Y_v = cargarDataset(path_base)
    W_Cholesky = fullyConnectedLineal_Cholesky(X_t, Y_t)
    W_QR = fullyConnectedLineal_QR(X_t, Y_t)
    print(f"W_QR: {W_QR}")
    W_SVD = fullyConnectedLineal_SVD(X_t, Y_t)
    print(f"W_SVD:{W_SVD}")
    return 

def res_tri_mat(Triang, Y, inferior=False):
    X = [] 

def fullyConnectedLineal_Cholesky(X, Y):
    #TODO
    filas,columnas = X.shape
    rango = get_rango(X)
    if (columnas==rango and filas>columnas):
        Xt_X = productoMatricial(traspuesta(X), X)
        L = calculaCholesky(Xt_X)
        U_cols = []
        for row in X: # = col in Xt
            U_cols.append(res_tri(L, row))
        U = traspuesta(np.array(U_cols))
        return productoMatricial(Y, U)
        return
    elif (rango == filas  and filas<columnas):
        X_Xt = productoMatricial(X,traspuesta(X))
        L = calculaCholesky(X_Xt)
        return 
    elif(rango == filas and filas==columnas):
        resultado = res_tri_mat(W,Y,True)
        return 
    else:
        #FIXME: se puede llegar a este caso?
        
        return None
    return None

def get_rango(X):
    _,Sigma,_ = svd_reducida(X)
    return len(Sigma)

def pinvEcuacionesNormales(X,L,Y):
    #TODO
    return None

def hermitiana(A):
    return traspuesta(A) #CORRECTO PARA DATOS REALES :D 
def esPseudoInversa(X,pX,tol=1e-8):
    # Con Moore Penrose
    X_pX = productoMatricial(X,pX)
    pX_X = productoMatricial(pX,X)
    X_pX_X = productoMatricial(X_pX,X)
    pX_X_pX = productoMatricial(pX,X_pX)
    # X_pX_X = X
    condicion_1 = matricesIguales(X_pX_X,X,atol=tol)
    if not condicion_1:return False
    # pX_X_pX = pX
    condicion_2 = matricesIguales(pX_X_pX,pX,atol=tol)
    if not condicion_2: return False
    # (X_pX)^* = X_pX
    condicion_3  = matricesIguales(hermitiana(X_pX),X_pX,atol=tol)
    if not condicion_3: return False
    condicion_4 = matricesIguales(hermitiana(pX_X),pX_X,atol=tol)
    if not condicion_4: return False
    return True 

#EVALUACIÓN Y BENCHMARKING 
#TODO: Generar una matriz de confusión evaluando a partir de los pares de embeddings de val o test (X_v,Y_v)
#TODO: Presentar una tabla comparativa de los resultados de cada metodología donde en las columnas se debe mostrar perf de clasificación 
#SINTESIS FINAL 
#TODO: 400 PALABRAS SINTETIZANDO




def list_to_diag(X):
    n = len(X)
    matriz_diagonal:np.ndarray = np.zeros((n,n))
    for i in range(n):
        matriz_diagonal[i][i]=X[i]
    return matriz_diagonal

def pinSVD(U,S,V,Y):
    X,Y = reducir_matrices_testeo(X,Y)
    U_traspuesta = traspuesta(U)
    S_inversa = list_to_diag(1.0 / S)  
    X_inversa = productoMatricial(productoMatricial(V,S_inversa),U_traspuesta)
    return productoMatricial(Y,X_inversa)
    
def fullyConnectedLineal_SVD(X:np.ndarray, Y:np.ndarray):
    n,_ = X.shape
    U_de_x,Sigma_de_x,V_de_x = svd_reducida(X,k=n)
    return pinSVD(U_de_x, Sigma_de_x, V_de_x,Y)

def fullyConnectedLineal_QR(X, Y):
    X, Y = reducir_matrices_testeo(X, Y)
    Q, R = QR_reducida(traspuesta(X))
    V = productoMatricial(Q,modulo_alc.inversa(traspuesta(R)))
    # V = pinv_v2(Q, R)
    return modulo_alc.productoMatricial(Y, V)

def reducir_matrices_testeo(X, Y):
    X = X[:,:10]
    Y = Y[:,:10]
    return X,Y


def QR_reducida(A, metodo='RH', tol=1e-12):
    Q, R = modulo_alc.calculaQR(modulo_alc.traspuesta(A), metodo, tol)
    return Q[:, :A.shape[1]], R[:A.shape[1],:]


def pinv_v2(Q, R):
    #V_rows = []
    #Qt = modulo_alc.traspuesta(Q)
    #cols_Qt = Qt.shape[1]
    #for i in range(cols_Qt):
    #    V_rows.append(modulo_alc.res_tri(R, Qt[:,i], False))
    #    print(V_rows[-1].shape)
    #V = np.array(V_rows)    
    #print(f"V.shape: {V.shape}")
    # return V
    print("hola")
    return

def pinvHouseHolder(Q,R,Y):
    #TODO
    return None 

def pinvGramSchmidt(Q,R,Y):
    #TODO
    return None

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
