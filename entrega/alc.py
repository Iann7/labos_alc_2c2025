import numpy as np

path_base = "./cats_and_dogs"

# region HELPERS

def esCuadrada(A):
    return A.shape[0] == A.shape[1]

def traspuesta(A):
    n, m = A.shape
    traspuesta = np.zeros((m, n))
    
    for i in range(n):
        for j in range(m):
            traspuesta[j][i] = A[i][j]

    return traspuesta

def esSimetrica(A, tol=0):
    if not esCuadrada(A):
        return False
    
    n, m = A.shape
    T = traspuesta(A)
    for i in range(n):
        for j in range(m):
            if not allclose(A[i][j], T[i][j], tol):
                return False
                
    return True

def calcularAx(A, x):
    #n, m = A.shape
    #b = np.zeros(n)
    #
    #for i in range(n):
    #    for j in range(m):
    #        b[i] += A[i][j] * x[j]
    return A @ x 

def productoEscalar(u, v):
    #u = np.array(u)
    #v = np.array(v)
    #n = u.shape[0]
    #if n != v.shape[0]:
    #    return None
    #suma = 0
    #for i in range(n):
    #    suma += u[i] * v[i]
    # Como nos dijo el corrector, usaremos las funciones de numpy para ahorrar tiempo
    return np.dot(np.array(u),np.array(v))

def determinanteTriangular(D):
    # Asumimos que la matriz es triangular
    if not esCuadrada(D):
        return None
    n, _ = D.shape
    res = 1
    for i in range(n):
        res *= D[i, i] 
    return res

def sumar_fila_multiplo(A, i, j, s):
    # Asumimos que i y j están en rango
    m = A.shape[1]
    for k in range(m):
        A[i][k] = A[i][k] + A[j][k] * s

def intercambiarFilas(A, i, j):
    # Asumimos que i y j están en rango
    m = A.shape[1]
    for k in range(m):
        tmp = A[i][k]
        A[i][k] = A[j][k]
        A[j][k] = tmp

def indiceMaxAbs(v):
    if v.size == 0:
        return -1
        
    maxInd = 0
    
    for i in range(v.size):
        if abs(v[i]) > abs(v[maxInd]):
            maxInd = i

    return maxInd
    
def inversaNoLU(A):
    n, m = A.shape
    if n != m: # or determinante(A) == 0:
        return None
    A = A.astype(float).copy()
    inversa = np.eye(n) # La inicializo como la identidad

    # Row echelon
    i = 0
    j = 0
    while i < n and j < m:
        ind = indiceMaxAbs(A[i:, j]) + i
        if A[ind][j] == 0:
            j += 1
            continue
        if i != ind:
            intercambiarFilas(A, i, ind)
            intercambiarFilas(inversa, i, ind)
        for k in range(i+1, n):
            escalar = -A[k][j]/A[i][j]
            sumar_fila_multiplo(A, k, i, escalar)
            sumar_fila_multiplo(inversa, k, i, escalar)
        i += 1
        j += 1

    # 1s en la diagonal
    for k in range(0, n):
        divisor = A[k][k]
        for h in range(0, n):
            A[k][h] = A[k][h] * 1/divisor
            inversa[k][h] = inversa[k][h] * 1/divisor

    # Diagonalizacion superior
    for k in range(n-1, 0, -1):
        for h in range(k-1, -1, -1):
            escalar = -A[h][k]
            sumar_fila_multiplo(A, h, k, escalar)
            sumar_fila_multiplo(inversa, h, k, escalar)

    return inversa

def filtrarMatriz(f, A):
    n, m = A.shape
    filtered = np.zeros((n, m))
    
    for i in range(n):
         for j in range(m):
             if f(i, j):
                 filtered[i][j] = A[i][j]

    return filtered

def triangSupConDiag(A):
    return filtrarMatriz(lambda i,j: i <= j, A)

def triangInfSinDiag(A):
    return filtrarMatriz(lambda i,j: i > j, A)

def sumarConIdentidad(A):
    Ac = A.copy()
    n = Ac.shape[0]
    for i in range(n):
        Ac[i,i] += 1
    return Ac

def productoMatricial(A, B):
    #A_n, A_m = A.shape
    #B_n, B_m = B.shape
    #assert A_m == B_n, "ERROR LAS DIMENSIONES NO MATCHEAN" 
    #res = np.zeros((A_n, B_m))
    #for i in range(A_n):
    #    for j in range(B_m):
    #        row_A = A[i]
    #        col_B = B[:, j]
    #        res[i, j] = productoEscalar(row_A, col_B)
    # Como nos dijo el corrector,usamos numpy para ahorrar tiempo
    return A @ B 
        
def matriz(v):
    if len(v.shape) > 1:
        return v
    return np.array([v])

def householder(v, u):
    n = v.shape[0]
    resta = u - v
    norm = (norma(resta, 2)**2)
    if norm < 1e-12:
        return np.eye(n)
    return np.eye(n) - 2 / norm * (productoMatricial(traspuesta(matriz(resta)), matriz(resta)))

def expandirMatriz(A, val, cant=1):
    n, m = A.shape
    res = np.eye(n + cant, m + cant)
    for i in range(n):
        for j in range(m):
            res[i + cant, j + cant] = A[i, j]

    for i in range(cant):
        res[i, i] = val

    return res

def allclose(x, y, atol=1e-8): # ! Consultar error relativo o absoluto
    return error(x, y) <= atol

# endregion

# region LABO 1
def error(x, y):
    return abs(np.float64(x) - np.float64(y))

def error_relativo(x, y):
    return abs(np.float64(x) - np.float64(y)) / abs(np.float64(x))

def matricesIguales(A, B, atol=1e-08):
    if A.shape != B.shape:
        return False
    n, m = A.shape

    for i in range(n):
        for j in range(m):
            if not allclose(A[i, j], B[i, j], atol):
                return False

    return True

# endregion

# region LABO 2

def rota(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    
def escala(s):
    return np.diag(s)

def rota_y_escala(theta, s):
    return escala(s) @ rota(theta)

def afin(theta, s, b):
    rye = rota_y_escala(theta,s)
    afinn = np.eye(3)
    afinn[:2, :2] = rye
    afinn[:2,  2] = b
    return afinn

def trans_afin(v, theta, s, b):
    v3 = np.ones(3)
    v3[:2] = v
    return (afin(theta, s, b) @ v3)[:2]

# endregion

# region LABO 3

def norma(x,p):
    if p == 'inf':
        max = -1
        for elem in x:
            if np.abs(elem) > max:
                max = np.abs(elem)
        return max

    suma = 0
    for elem in x:
        suma += np.abs(elem)**p
    return suma**(1.0/p)

def normaliza(X,p):
    res = []
    for x in X:
        res.append(x*(np.float64(1/norma(x,p))))
    return res

def normaMatMC(A,q,p,Np):
    maxNorma = -1
    vec = []
    tamanio = A.shape[1]
    for _ in range(Np):
        x = normaliza([np.random.rand(tamanio)], p)[0]
        # if not np.allclose(norma(x,p), 1):
            # print("ERROR NORMA")
        alpha = calcularAx(A,x)
        normaAlpha = norma(alpha, q)
        if normaAlpha > maxNorma:
            vec = alpha
            maxNorma = normaAlpha


    # print(f"Norma de vector Ax con max norma: {maxNorma}")
    # print(f"Vector Ax con max norma: {vec}")
    return [maxNorma, vec]

def normaExacta(A,p=[1,'inf']):
    m, n = A.shape
    if p == ['inf'] or p == 'inf':
        maxInf = -1
        for i in range(m):
            sumaPorFila = 0
            fila = A[i]
            for j in range(n):
                sumaPorFila += abs(fila[j])
            maxInf = max(maxInf, sumaPorFila)
        return maxInf
    elif p == [1] or p == 1:
        maxOne = -1
        for j in range(n):
            sumaPorCol = 0
            for i in range(m):
                sumaPorCol += abs(A[i][j])
            maxOne = max(maxOne, sumaPorCol)
        return maxOne
    return None


def condMC(A, p, Np):
    return normaMatMC(A, p, p, Np)[0] * normaMatMC(inversaNoLU(A), p, p, Np)[0]

def condExacta(A, p):
    return normaExacta(A, p) * normaExacta(inversaNoLU(A), p)

# endregion

# region LABO 4

def calculaLU(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        return None, None, 0

    for k in range(n-1):
        if Ac[k,k] == 0: 
            return None, None, 0
        for i in range(k+1, n):
            m_i = Ac[i,k]/Ac[k,k]
            cant_op += 1
            for j in range(k+1, n):
                Ac[i,j] = Ac[i,j] - m_i*Ac[k,j]
                cant_op += 2    
            Ac[i,k] = m_i 
            
    L = obtenerL(Ac)
    U = obtenerU(Ac)
    # print(f"L: {L}")
    # print(f"U: {U}")
    

    return L, U, cant_op

def obtenerL(A):
    return sumarConIdentidad(triangInfSinDiag(A))

def obtenerU(A):
    return triangSupConDiag(A)

def res_tri(L, b, inferior=True):
    # asumimos que viene triang sup o inferior
    n, m = L.shape
    if m!=n:
        return None
    
    x = np.zeros(n)
    
    if inferior:
        for i in range(n):
            if i == 0:
                x[i] = b[i]/L[i,i]
            else: 
                suma = 0
                for j in range(i):
                    suma += L[i,j]*x[j]
                x[i] = (b[i] - suma)/L[i,i]
        return x
    else:
        for i in range(n-1, -1, -1):
            if i == n-1:
                x[i] = b[i]/L[i,i]
            else: 
                suma = 0
                for j in range(i+1, n):
                    suma += L[i,j]*x[j]
                x[i] = (b[i] - suma)/L[i,i]
        return x
    

def inversa(A):
    assert esCuadrada(A), "No es cuadrada y se intento hacer inversa"
    L, U, _ = calculaLU(A)
    assert determinanteTriangular(U) != 0, "Determinante = 0, se intento hacer inversa"
    n, _ = A.shape
    inv = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1
        y = res_tri(L, e, True)
        x = res_tri(U, y, False)
        for i in range(n):
            inv[i, j] = x[i]
    return inv

def calculaLDV(A): 
    if not esCuadrada(A):
        return None, None, None, 0
    
    nops = 0
    
    L, U, nopsLU = calculaLU(A)

    nops += nopsLU
    
    U_t = traspuesta(U)
    
    V_t, D, nopsVtD = calculaLU(U_t)

    nops += nopsVtD
    
    V = traspuesta(V_t)
    return L, D, V, nopsVtD

def esSDP(A, atol=1e-8):
    if not esSimetrica(A):
        return False
    _, D, _, _ = calculaLDV(A)
    if D is None:
        return False
    n = D.shape[0]
    for i in range(n):
        if not (D[i, i] > atol):
            return False

    return True

def calculaCholesky(A, atol=1e-10):
    # Asumimos que el A es sdp para evitar operaciones innecesarias y errores por tolerancia
    # if not esSDP(A, atol):
    #     return None
    n, _ = A.shape
    L = np.zeros((n, n))
    for k in range(n):
        L[k, k] = A[k, k]
        for j in range(k):
            L[k, k] -= pow(L[k, j],2)
        L[k, k] = np.sqrt(L[k, k])
        for i in range(k+1, n):
            L[i, k] = A[i, k]
            for j in range(k):
                L[i, k] -= L[i, j] * L[k, j]
            L[i, k] *= 1 / L[k, k]

    return L

# endregion

# region LABO 5

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    m, n = A.shape

    norma2 = norma(A[:,0], 2) 
    qs = [A[:,0] * 1/norma2] # q_1 = a_1/||a_1||_2
    rs = [[norma2]] # r_11 = ||a_1||_2
    nops = 0

    for j in range(1, n):

        q_j = A[:,j] # q_j = a_j

        for k in range(len(qs)):
            r_kj = productoEscalar(qs[k], q_j) # r_kj = q_k^t * q_j
            nops += 2 * n - 1# n sumas y n multiplicaciones
            rs[k].append(r_kj) 
            q_j = q_j - (qs[k] * r_kj) # q_j = q_j - r_kj * q_k
            nops += 2

        norma2 = norma(q_j, 2) # Hay que contar las operaciones?
        rs.append([norma2]) # r_jj = ||q_j||_2

        if allclose(norma2, 0, tol):
            continue

        qs.append(q_j * (1/norma2)) # true_q_j = q_j / r_jj
        nops += 2

    Q = traspuesta(np.array(qs))
    R = np.zeros((len(qs), n))

    # Terminamos de rellenar con 0s R
    for i in range(len(qs)):
        for j in range(len(rs[i])):
            R[i, i + j] = rs[i][j]
            
    if retorna_nops:
        return Q, R, nops
    return Q, R

def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    m, n = A.shape
    
    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        x = R[k:m, k].copy()
        norma2 = norma(x, 2)
        if norma2 <= tol:
            continue
        e_1 = np.zeros(m - k)
        e_1[0] = norma2
        H_k = householder(x, e_1)
        H_k_ = expandirMatriz(H_k, 1, k)
        R = productoMatricial(H_k_, R)
        Q = productoMatricial(Q, traspuesta(H_k_))

    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    return None

# endregion

# region LABO 6

def f(A, v):
    n, _ = A.shape
    wp = calcularAx(A, v)
    norma2 = norma(wp, 2)
    if allclose(norma2, 0):
        return np.zeros(n)
    else:
        return wp * (1/norma2)
    

def metpot2k(A,tol=1e-8,K=100):
    n, m = A.shape
    if n != m:
        return None
    
    v = np.random.randn(n)
    v_ = f(A, v)
    v_ = f(A, v_)
    e = productoEscalar(v_, v)
    k = 0
    while np.abs(e-1) > tol and k < K:
        v = v_
        v_ = f(A, v)
        v_ = f(A, v_)
        e = productoEscalar(v_, v)
        k = k + 1

    autovalor = productoEscalar(v_, calcularAx(A, v_))
    err = e - 1
    return v_, autovalor, k


def diagRH(A,tol=1e-15,K=1000):
    # Forzamos que sea simetrica
    A = (A + traspuesta(A)) / 2
    # No checkeamos que sea simetrica para evitar costo de operaciones en el TP
    # y para evitar arrastrar errores de precision al hacer tantas cuentas
    # if not esSimetrica(A, 100*tol):
    #     print("FALLO POR NO SIMETRICA")
    #     return None
    n, _ = A.shape
    v_1, val_1, _ = metpot2k(A, tol, K)
    e_1 = np.zeros(n)
    e_1[0] = 1
    H_v_1 = householder(e_1, v_1)
    if n == 2:
        S = H_v_1
        D = productoMatricial(H_v_1, productoMatricial(A, traspuesta(H_v_1)))
    else:
        B = productoMatricial(H_v_1, productoMatricial(A, traspuesta(H_v_1)))
        A_ = B[1:n, 1:n]
        S_, D_ = diagRH(A_, tol, K)
        D = expandirMatriz(D_, val_1)
        S = productoMatricial(H_v_1, expandirMatriz(S_, 1))
    return S, D

# endregion

# region LABO 7

def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    T = []
    for i in range(n):
        vec = normaliza([np.random.rand(n)], 1)[0]
        # print(f"vec: {vec}")
        T.append(vec)
    return traspuesta(np.array(T))


def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    T = transiciones_al_azar_continuas(n)
    # print(f"T: {T}")

    for i in range(n):
        for j in range(n):
            if T[i, j] <= thres:
                T[i, j] = 1
            else:
                T[i, j] = 0

    for i in range(n):
        col = T[:, i]
        if allclose(norma(col, 1), 0):
            fila = np.random.randint(0, n)
            T[fila, i] = 1
        
    # print(f"T: {T}")
    res_t = np.array(normaliza(traspuesta(T), 1))
    # print(f"res_t: {res_t}")
    res = traspuesta(res_t)
    # print(f"res: {res}")
    return res

def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    AA = productoMatricial(traspuesta(A), A)
    # print(f"A: {A}")
    # print(f"AA: {AA}")
    n, _ = AA.shape
    S, D = diagRH(A)
    # print(f"S: {S}")
    # print(f"D: {D}")
    res = []
    for i in range(n):
        if allclose(D[i, i], 0, tol):
            res.append(S[:, i])

    # print(f"res: {res}")
    return traspuesta(matriz(np.array(res)))
        

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    res = {}

    if len(listado) > 0:
        I, J, A_ij = listado
        if len(I) != len(J) or len(I) != len(A_ij):
            return None
        for k in range(len(I)):
            if not allclose(A_ij[k], 0):
                res[(I[k], J[k])] = A_ij[k]

    return res, (m_filas, n_columnas)
        

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    M, (n, m) = A
    if m != v.shape[0]:
        return None

    res = np.zeros(n)
    for k in range(n):
        keys_k = [(i, j) for (i, j) in M if i == k] # obtengo las keys de la fila i
        for (i, j) in keys_k:
            res[i] += M[(i, j)] * v[j]

    return res
            
            
# endregion

# region LABO 8
def calcular_sigma_casita(A,tol):
    sigma_casita = []
    for i in range(A.shape[0]):
        if A[i][i] > tol:    
            sigma_casita.append(np.sqrt(A[i][i]))
        else:
            break
    return sigma_casita

def svd_reducida(A,k="max",tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """
    filas_a,columnas_a= A.shape
    a_t_a = productoMatricial(traspuesta(A),A)
    # Nos aseguramos de que sea simetrica con este (M + Mt) / 2
    a_t_a = (a_t_a + traspuesta(a_t_a)) / 2
    V,diagonal_autovalores_de_ata = diagRH(a_t_a,tol)
    Sigma_casita_diagonal = calcular_sigma_casita(diagonal_autovalores_de_ata , tol)
    r=len(Sigma_casita_diagonal)
    V_casita = V[:,:r]
    B = productoMatricial(A, V_casita)  
    U_t_norm = np.array(normaliza(traspuesta(B),p=2))
    U_casita = traspuesta(U_t_norm)
    if k == "max":
        k = r
    
    return  U_casita[:,:k],np.array(Sigma_casita_diagonal)[:k],V_casita[:,:k]


# endregion

# region Helpers TP
def res_tri_mat(Triang, Y, inferior=True):
    """
    Output: 
        X solucion del sistema triangular.
    Args:
        Triang (np.ndarray): Matriz triangular (superior o inferior).
        Y (np.ndarray): Matriz de terminos independientes.
        inferior (bool, optional): Booleano que indica si la matriz es inferior.
    """
    X_cols = [] 
    Y_cols_count = Y.shape[1]
    for i in range(Y_cols_count):
        X_cols.append(res_tri(Triang, Y[:, i], inferior))
    X = traspuesta(np.array(X_cols))
    return X

def hermitiana(A):
    """
    Output: 
        Matriz hermitiana (traspuesta para matrices reales).
    Args:
        A (np.ndarray): Matriz de entrada.
    """
    return traspuesta(A) #CORRECTO PARA DATOS REALES :D 

def list_to_diag(X):
    """
    Output: 
        Matriz diagonal con los elementos de X en la diagonal.
    Args:
        X (list/np.ndarray): Lista o vector con los elementos de la diagonal.
    """
    n = len(X)
    matriz_diagonal:np.ndarray = np.zeros((n,n))
    for i in range(n):
        matriz_diagonal[i][i]=X[i]
    return matriz_diagonal

def reducir_matrices_testeo(X, Y, cant=100):
    """
    Output: 
        Se devuelven X e Y reducidas cantidad cant de perros y cantidad cant de gatos.
    Args:
        X (np.ndarray): Matriz de embeddings (tamaño mayor a cant).
        Y (np.ndarray): Matriz de targets (tamaño mayor a cant).
        cant (int, optional): Cantidad de perros y gatos a tener en la matriz final
    """
    if cant > 0 : # No reducir si es menor o igual que 0
        print(f"Reduciendo matrices de entrenamiento a {cant} ejemplos por clase")
        X_dogs = X[:,:cant] 
        Y_dogs = Y[:,:cant]
        # voy a agarrar las ultimas diez entradas de X que son los cats
        X_cats = X[:,-cant:]
        Y_cats = Y[:,-cant:]
        X = np.column_stack((X_dogs, X_cats))
        Y = np.column_stack((Y_dogs, Y_cats))
        
    return X,Y

def invertirDiagonal(D):
    """
    Output: 
        Inversa de la matriz diagonal D. None si no es inversible (Si la matriz tiene numeros != 0 fuera de la diagonal, se asumen 0).
    Args:
        D (np.ndarray): Matriz diagonal.
    """
    if not esCuadrada(D):
        return None

    n, _ = D.shape
    inv = np.zeros((n, n))

    for i in range(n):
        if allclose(D[i, i], 0):
            return None
        inv[i, i] = 1.0 / D[i, i]

    return inv

def QR_reducida(A, metodo='RH', tol=1e-12):
    """
    Output: 
        Matrices Q y R de la descomposicion QR de A en su version reducida.
    Args:
        A (np.ndarray): Matriz de entrada.
        metodo (str, optional): Metodo para descomponer en QR ('RH' o 'GS').
        tol (float, optional): Tolerancia.
    """
    _, cols = A.shape
    Q, R = calculaQR(A, metodo, tol)
    return reducirQR(Q, R)

def reducirQR(Q, R):
    """
    Asumimos que la At = QR tiene mas columnas que filas. Esto lo hacemos porque en el mismo pseudocodigo del algoritmo 3 se aclara que estamos en ese caso.

    Output: 
        Matrices Q y R reducidas.
    Args:
        Q (np.ndarray): Matriz ortogonal Q.
        R (np.ndarray): Matriz triangular R.
    """
    m, n = R.shape
    
    k = n
    
    Q_red = Q[:, :k]
    R_red = R[:k, :]
    
    return Q_red, R_red

def generarY(vector, n):
    """
    Output: 
        Matriz Y con el vector repetido n veces como columnas.
    Args:
        vector (np.ndarray): Vector de [0,1] o [1,0].
        n (int): Cantidad de columnas.
    """
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
# endregion

# region Funciones TP
def cargarDataset(carpeta):
    """
    Output: 
        Tupla con cuatro matrices (X_t, Y_t, X_v, Y_v) correspondientes a embeddings y targets de entrenamiento y validación.
    Args:
        carpeta (str): Path al directorio del dataset.
    """
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

# region Algoritmo 1

def fullyConnectedLineal_Cholesky(X:np.ndarray, Y:np.ndarray):
    """
    Output: 
        Matriz de pesos W calculada usando descomposición de Cholesky.
    Args:
        X (np.ndarray): Matriz de embeddings de entrenamiento.
        Y (np.ndarray): Matriz de targets de entrenamiento.
    """
    #X, Y = reducir_matrices_testeo(X, Y)
    cant_filas_X,cant_columnas_X = X.shape
    Xt = traspuesta(X)
    W = None
    # No calculamos el rango porque asumimos que X es de rango completo por precondicion
    if cant_filas_X>=cant_columnas_X:
        Xt_X_con_cholesky = productoMatricial(Xt, X)
        L = calculaCholesky(Xt_X_con_cholesky)
        W = pinvEcuacionesNormales(X, L, Y)

    elif cant_filas_X<cant_columnas_X:
        X_Xt = productoMatricial(X, Xt)
        L = calculaCholesky(X_Xt)
        W = pinvEcuacionesNormales(X, L, Y)

    return W

def pinvEcuacionesNormales(X, L, Y):
    """
    Output: 
        Matriz de pesos W.
    Args:
        X (np.ndarray): Matriz de embeddings.
        L (np.ndarray): Matriz triangular resultante de la descomposición de Cholesky.
        Y (np.ndarray): Matriz de targets.
    """
    filas, columnas = X.shape
    W = None
    if filas >= columnas:
        # Caso filas == columnas vale el caso a) por lo siguiente:
        # X+ = (Xt * X)^-1 * Xt
        # X+ = X^-1 * Xt^-1 * Xt
        # X+ = X^-1 * I
        # X+ = X^-1
        # Entonces WX = Y es lo mismo que W = Y(X^-1) = YU

        # L = Cholesky(Xt * X)
        # L * Lt * U = Xt
        # Resolvemos para U:
        # L * A = Xt
        # Lt * U = A

        # L * A = Xt
        A = res_tri_mat(L, traspuesta(X))

        # Lt * U = A
        U = res_tri_mat(traspuesta(L), A, inferior=False)

        # W = Y * U
        W = productoMatricial(Y, U)
    
    elif filas < columnas:
        # L = Cholesky(X * Xt)
        # V * (L * Lt) = Xt
        # (V * (L * Lt))t = X
        # L * Lt * Vt = X

        # Entonces Resolvemos para Vt:
        # L * (Lt * Vt) = X
        # L * A = X
        # Lt * Vt = A

        # L * A = X
        A = res_tri_mat(L, X)

        # Lt * Vt = A
        Vt = res_tri_mat(traspuesta(L), A, inferior=False)

        V = traspuesta(Vt)
        W = productoMatricial(Y, V)
    
    # Asumimos que deberíamos caer en alguno de los dos casos anteriores
    return W

# endregion

# region Algoritmo 2

def fullyConnectedLineal_SVD(X:np.ndarray, Y:np.ndarray):
    """
    Output: 
        Matriz de pesos W calculada usando descomposición SVD.
    Args:
        X (np.ndarray): Matriz de embeddings de entrenamiento.
        Y (np.ndarray): Matriz de targets de entrenamiento.
    """
    X, Y = reducir_matrices_testeo(X, Y)
    n,_ = X.shape
    U_de_x,Sigma_de_x,V_de_x = svd_reducida(X,k=n)
    return pinvSVD(U_de_x, list_to_diag(Sigma_de_x), V_de_x,Y)

def pinvSVD(U:np.ndarray,S:np.ndarray,V:np.ndarray,Y:np.ndarray):
    """
    Output: 
        Matriz de pesos W.
    Args:
        U (np.ndarray): Matriz U de la descomposición SVD.
        S (np.ndarray): Matriz diagonal de valores singulares.
        V (np.ndarray): Matriz V de la descomposición SVD.
        Y (np.ndarray): Matriz de targets.
    """
    if S.ndim == 1: # En caso de que S venga como una lista, armamos la matriz
        S = list_to_diag(S)
    Ur, Sr, Vr = reducirSVD(U, S, V)
    Ur_traspuesta = traspuesta(Ur)
    Sr_inversa = invertirDiagonal(Sr)
    print(f"Vr {Vr.shape}")
    print(f"Sr_inversa {Sr_inversa.shape}")
    VrSr = productoMatricial(Vr,Sr_inversa)
    print(f"VrSr {VrSr.shape}")
    print(f"UrT {Ur_traspuesta.shape}")
    X_inversa = productoMatricial(VrSr, Ur_traspuesta)
    return productoMatricial(Y, X_inversa)

def reducirSVD(U:np.ndarray, S:np.ndarray, V:np.ndarray, atol=1e-10):
    """
    Asumimos que S tiene algun valor singular no nulo

    Output: 
        Matrices U, S, V reducidas eliminando valores singulares cercanos a cero.
    Args:
        U (np.ndarray): Matriz U de la SVD.
        S (np.ndarray): Matriz S de la SVD.
        V (np.ndarray): Matriz V de la SVD.
        atol (float, optional): Tolerancia para considerar un valor singular como cero.
    """
    cantidad_de_rs = 0
    print(S.shape)
    n = min(S.shape[0], S.shape[1])
    for i in range(n):
        # print(f"S[{i}] = {S[i, i]}")
        if allclose(0, S[i,i], atol):
            break
        cantidad_de_rs += 1

    print(f"Cantidad de rs: {cantidad_de_rs}")

    return U[:, :cantidad_de_rs], S[:cantidad_de_rs, :cantidad_de_rs], V[:cantidad_de_rs, :] 

# endregion

# region Algoritmo 3

def fullyConnectedLineal_QR(X:np.ndarray, Y:np.ndarray, metodo='GS'):
    """
    Output: 
        Matriz de pesos W calculada usando descomposición QR.
    Args:
        X (np.ndarray): Matriz de embeddings de entrenamiento.
        Y (np.ndarray): Matriz de targets de entrenamiento.
        metodo (str, optional): Método para QR ('GS' o 'RH').
    """
    #X, Y = reducir_matrices_testeo(X, Y)
    Q, R = QR_reducida(traspuesta(X), metodo)
    W = pinvHouseHolder(Q, R, Y)
    return W


def pinvHouseHolder(Q:np.ndarray, R:np.ndarray, Y:np.ndarray):    
    """
    Da igual el metodo utilizado, dado a que se la pasa la Q,R ya factorizada

    Output: 
        Matriz de pesos W.
    Args:
        Q (np.ndarray): Matriz ortogonal Q.
        R (np.ndarray): Matriz triangular superior R.
        Y (np.ndarray): Matriz de targets.
    """
    Qr, Rr = reducirQR(Q, R)
    V_t = res_tri_mat(Rr, traspuesta(Qr), inferior=False)
    V = traspuesta(V_t)
    W = productoMatricial(Y, V)
    return W

def pinvGramSchmidt(Q, R, Y):
    """
    Da igual el metodo utilizado, dado a que se la pasa la Q,R ya factorizada

    Output: 
        Matriz de pesos W.
    Args:
        Q (np.ndarray): Matriz ortogonal Q.
        R (np.ndarray): Matriz triangular superior R.
        Y (np.ndarray): Matriz de targets.
    """
    return pinvHouseHolder(Q, R, Y)

# endregion

def esPseudoInversa(X,pX,tol=1e-8):
    """
    Output: 
        Booleano indicando si pX es efectivamente la pseudo-inversa de Moore-Penrose de X.
    Args:
        X (np.ndarray): Matriz original.
        pX (np.ndarray): Matriz candidata a pseudo-inversa.
        tol (float, optional): Tolerancia para las comparaciones numéricas.
    """
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

# endregion

def main():
    # Esta seccion la utilizamos para debugging
    return 

if __name__ == "__main__":
    main()