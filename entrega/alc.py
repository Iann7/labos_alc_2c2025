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
        e_1[0] = 1
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

# region TESTS Labos

def tests1():

    def sonIguales(x, y, atol=1e-08):
        return np.allclose(error(x, y), 0, atol=atol)

    assert(not sonIguales(1,1.1))
    assert(sonIguales(1,1 + np.finfo('float64').eps))
    assert(not sonIguales(1,1 + np.finfo('float32').eps))
    assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
    assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))

    assert(np.allclose(error_relativo(1,1.1),0.1))
    assert(np.allclose(error_relativo(2,1),0.5))
    assert(np.allclose(error_relativo(-1,-1),0))
    assert(np.allclose(error_relativo(1,-1),2))

    assert(matricesIguales(np.diag([1,1]),np.eye(2)))
    assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
    assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

def tests2():
    # Tests para rota
    assert(np.allclose(rota(0), np.eye(2)))
    assert(np.allclose(rota(np.pi/2), np.array([[0, -1],[1, 0]])))
    assert(np.allclose(rota(np.pi), np.array([[-1, 0],[0, -1]])))
    
    # Tests para escala
    assert(np.allclose(escala([2,3]), np.array([[2,0],[0,3]])))
    assert(np.allclose(escala([1,1,1]), np.eye(3)))
    assert(
        np.allclose(escala([0.5,0.25]), np.array([[0.5,0],[0,0.25]]))
    )
    
    # Tests para rota_y_escala
    assert(
        np.allclose(rota_y_escala(0,[2,3]), np.array([[2,0],[0,3]]))
    )
    
    assert(np.allclose(
        rota_y_escala(np.pi/2,[1,1]), np.array([[0,-1],[1,0]])
    ))
    
    assert(np.allclose(
        rota_y_escala(np.pi,[2,2]), np.array([[-2,0],[0,-2]]))
    )
    
    # Tests para afin
    assert(np.allclose(
        afin(0,[1,1],[1,2]),
        np.array([[1,0,1],
                  [0,1,2],
                  [0,0,1]])
    ))
    
    assert(np.allclose(
        afin(np.pi/2,[1,1],[0,0]),
        np.array([[0,-1,0],
                  [1, 0,0],
                  [0, 0,1]])
    ))
    
    assert(np.allclose(
        afin(0,[2,3],[1,1]),
        np.array([[2,0,1],
                  [0,3,1],
                  [0,0,1]])
    ))
    
    # Tests para trans_afin
    assert(np.allclose(
        trans_afin(np.array([1,0]), np.pi/2,[1,1],[0,0]),
        np.array([0,1])
    ))
    assert(np.allclose(
        trans_afin(np.array([1,1]), 0,[2,3],[0,0]),
        np.array([2,3])
    ))
    assert(np.allclose(
        trans_afin(np.array([1,0]), np.pi/2,[3,2],[4,5]),
        np.array([4,7])
    ))

def tests3():
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


    # Tests normaExacta

    assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
    assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
    assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
    assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
    assert(normaExacta(np.random.random((10,10)),1)<=10)
    assert(normaExacta(np.random.random((4,4)),'inf')<=4)

    # Test normaMC

    nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
    assert(np.allclose(nMC[0],1,atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

    nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
    assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

    A = np.array([[1,2],[3,4]])
    nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
    # print(f"nMC[0]: {nMC[0]}")
    # print(f"norma exacta: {normaExacta(A,'inf')}")
    assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

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

def tests4():
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
    print(f"nops = {nops}")
    assert(nops == 13)

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
    L,D,V,nops = calculaLDV(A)
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
    print(f"V0: {V0}")
    A = L0 @ D0 @ V0
    print(f"A: {A}")
    assert(not esSDP(A))

def tests5():
    A2 = np.array([[1., 2.],
                [3., 4.]])

    A3 = np.array([[1., 0., 1.],
                [0., 1., 1.],
                [1., 1., 0.]])

    A4 = np.array([[2., 0., 1., 3.],
                [0., 1., 4., 1.],
                [1., 0., 2., 0.],
                [3., 1., 0., 2.]])

    # --- Funciones auxiliares para los tests ---
    def check_QR(Q,R,A,tol=1e-10):
        # Comprueba ortogonalidad y reconstrucción
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
        assert np.allclose(Q @ R, A, atol=tol)

    # --- TESTS PARA QR_by_GS2 ---
    Q2,R2 = QR_con_GS(A2)
    check_QR(Q2,R2,A2)

    Q3,R3 = QR_con_GS(A3)
    check_QR(Q3,R3,A3)

    Q4,R4 = QR_con_GS(A4)
    check_QR(Q4,R4,A4)

    # --- TESTS PARA QR_by_HH ---
    Q2h,R2h = QR_con_GS(A2)
    check_QR(Q2h,R2h,A2)

    Q3h,R3h = QR_con_HH(A3)
    check_QR(Q3h,R3h,A3)

    Q4h,R4h = QR_con_HH(A4)
    check_QR(Q4h,R4h,A4)

    # --- TESTS PARA calculaQR ---
    Q2c,R2c = calculaQR(A2,metodo='RH')
    check_QR(Q2c,R2c,A2)

    Q3c,R3c = calculaQR(A3,metodo='GS')
    check_QR(Q3c,R3c,A3)

    Q4c,R4c = calculaQR(A4,metodo='RH')
    check_QR(Q4c,R4c,A4)
    
def tests6():
    # Tests metpot2k
    S = np.vstack([
        np.array([2,1,0])/np.sqrt(5),
        np.array([-1,2,5])/np.sqrt(30),
        np.array([1,-2,1])/np.sqrt(6)
                ]).T

    # Pedimos que pase el 95% de los casos
    exitos = 0
    for i in range(100):
        D = np.diag(np.random.random(3)+1)*100
        A = S@D@S.T
        v,l,_ = metpot2k(A,1e-15,1e5)
        if np.abs(l - np.max(D))< 1e-8:
            exitos += 1
    assert exitos > 95


    #Test con HH
    exitos = 0
    for i in range(100):
        v = np.random.rand(9)
        #v = np.abs(v)
        #v = (-1) * v
        ixv = np.argsort(-np.abs(v))
        D = np.diag(v[ixv])
        I = np.eye(9)
        H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

        A = H@D@H.T
        v,l,_ = metpot2k(A, 1e-15, 1e5)
        #max_eigen = abs(D[0][0])
        if abs(l - D[0,0]) < 1e-8:         
            exitos +=1
    assert exitos > 95



    # Tests diagRH
    D = np.diag([1,0.5,0.25])
    S = np.vstack([
        np.array([1,-1,1])/np.sqrt(3),
        np.array([1,1,0])/np.sqrt(2),
        np.array([1,-1,-2])/np.sqrt(6)
                ]).T

    A = S@D@S.T
    SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
    assert np.allclose(D,DRH)
    assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



    # Pedimos que pase el 95% de los casos
    exitos = 0
    for i in range(100):
        A = np.random.random((5,5))
        A = 0.5*(A+A.T)
        S,D = diagRH(A,tol=1e-15,K=1e5)
        ARH = S@D@S.T
        e = normaExacta(ARH-A,p='inf')
        if e < 1e-5: 
            exitos += 1
    assert exitos >= 95

def tests7():
    def es_markov(T,tol=1e-6):
        """
        T una matriz cuadrada.
        tol la tolerancia para asumir que una suma es igual a 1.
        Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
        """
        n = T.shape[0]
        for i in range(n):
            for j in range(n):
                if T[i,j]<0:
                    return False
        for j in range(n):
            suma_columna = sum(T[:,j])
            if np.abs(suma_columna - 1) > tol:
                return False
        return True

    def es_markov_uniforme(T,thres=1e-6):
        """
        T una matriz cuadrada.
        thres la tolerancia para asumir que una entrada es igual a cero.
        Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
        """
        if not es_markov(T,thres):
            return False
        # cada columna debe tener entradas iguales entre si o iguales a cero
        m = T.shape[1]
        for j in range(m):
            non_zero = T[:,j][T[:,j] > thres]
            # all close
            close = all(np.abs(non_zero - non_zero[0]) < thres)
            if not close:
                return False
        return True

    def esNucleo(A,S,tol=1e-5):
        """
        A una matriz m x n
        S una matriz n x k
        tol la tolerancia para asumir que un vector esta en el nucleo.
        Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
        """
        for col in S.T:
            res = A @ col
            if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
                return False
        return True

    ## TESTS
    # transiciones_al_azar_continuas
    # transiciones_al_azar_uniformes
    for i in range(1,100):
        T = transiciones_al_azar_continuas(i)
        assert es_markov(T), f"transiciones_al_azar_continuas fallo para n={i}"
        
        T = transiciones_al_azar_uniformes(i,0.3)
        assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
        # Si no atajan casos borde, pueden fallar estos tests. Recuerden que suma de columnas DEBE ser 1, no valen columnas nulas.
        T = transiciones_al_azar_uniformes(i,0.01)
        assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
        T = transiciones_al_azar_uniformes(i,0.01)
        assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
        
    # nucleo
    A = np.eye(3)
    S = nucleo(A)
    assert S.shape[0]==0, "nucleo fallo para matriz identidad"
    A[1,1] = 0
    S = nucleo(A)
    msg = "nucleo fallo para matriz con un cero en diagonal"
    assert esNucleo(A,S), msg
    assert S.shape==(3,1), msg
    assert abs(S[2,0])<1e-2, msg
    assert abs(S[0,0])<1e-2, msg

    v = np.random.random(5)
    v = v / np.linalg.norm(v)
    H = np.eye(5) - np.outer(v, v)  # proyección ortogonal
    S = nucleo(H)
    msg = "nucleo fallo para matriz de proyeccion ortogonal"
    assert S.shape==(5,1), msg
    v_gen = S[:,0]
    v_gen = v_gen / np.linalg.norm(v_gen)
    assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg

    # crea rala
    listado = [[0,17],[3,4],[0.5,0.25]]
    A_rala_dict, dims = crea_rala(listado,32,89)
    assert dims == (32,89), "crea_rala fallo en dimensiones"
    assert A_rala_dict[(0,3)] == 0.5, "crea_rala fallo"
    assert A_rala_dict[(17,4)] == 0.25, "crea_rala fallo"
    assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

    listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
    A_rala_dict, dims = crea_rala(listado,50,50)
    assert dims == (50,50), "crea_rala fallo en dimensiones con tol"
    assert A_rala_dict.get((32,3)) == 7
    assert A_rala_dict[(16,4)] == 0.5
    assert A_rala_dict[(5,7)] == 0.25

    listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
    A_rala_dict, dims = crea_rala(listado,10,10)
    assert dims == (10,10), "crea_rala fallo en dimensiones con tol"
    assert (1,4) not in A_rala_dict
    assert A_rala_dict[(2,5)] == 0.5
    assert A_rala_dict[(3,6)] == 0.25
    assert len(A_rala_dict) == 2

    # caso borde: lista vacia. Esto es una matriz de 0s
    listado = []
    A_rala_dict, dims = crea_rala(listado,10,10)
    assert dims == (10,10), "crea_rala fallo en dimensiones con lista vacia"
    assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia"

    # multiplica rala vector
    listado = [[0,1,2],[0,1,2],[1,2,3]]
    A_rala = crea_rala(listado,3,3)
    v = np.random.random(3)
    v = v / np.linalg.norm(v)
    res = multiplica_rala_vector(A_rala,v)
    A = np.array([[1,0,0],[0,2,0],[0,0,3]])
    res_esperado = A @ v
    assert np.allclose(res,res_esperado), "multiplica_rala_vector fallo"

    A = np.random.random((5,5))
    A = A * (A > 0.5) 
    listado = [[],[],[]]
    for i in range(5):
        for j in range(5):
            listado[0].append(i)
            listado[1].append(j)
            listado[2].append(A[i,j])
            
    A_rala = crea_rala(listado,5,5)
    v = np.random.random(5)
    assert np.allclose(multiplica_rala_vector(A_rala,v), A @ v)

def tests8():
    # Tests L08    
    # Matrices al azar
    def genera_matriz_para_test(m,n=2,tam_nucleo=0):
        if tam_nucleo == 0:
            A = np.random.random((m,n))
        else:
            A = np.random.random((m,tam_nucleo))
            A = np.hstack([A,A])
        return(A)

    def test_svd_reducida_mn(A,tol=1e-15):
        m,n = A.shape
        print(A)
        hU,hS,hV = svd_reducida(A,tol=tol)
        nU,nS,nVT = np.linalg.svd(A)
        r = len(hS)+1
        #k el numero de valores singulares (y vectores) a retener.
        assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
        assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
        assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
        assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

    for m in [2,5,10,20]:
        for n in [2,5,10,20]:
            for i in range(2):
                A = genera_matriz_para_test(m,n)
                test_svd_reducida_mn(A)


    # Matrices con nucleo

    m = 12
    for tam_nucleo in [2,4,6]:
        for _ in range(10):
            A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
            test_svd_reducida_mn(A)

    # Tamaños de las reducidas
    A = np.random.random((8,6))
    for k in [1,3,5]:
        hU,hS,hV = svd_reducida(A,k=k)
        assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
        assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
        assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
        assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
        assert len(hS) == k, 'Tamaño de hS incorrecto'

def testsLabo():
    print('Ejecutando tests labo 1')
    #tests1()
    print('Ejecutando tests labo 2')
    #tests2()
    print('Ejecutando tests labo 3')
    #tests3()
    print('Ejecutando tests labo 4')
    # tests4()
    # A = np.array([
    #     [4.0, 12.0, -16.0],
    #     [12.0, 37.0, -43.0],
    #     [-16.0, -43.0, 98.0]
    # ], dtype=float)
    # print(f"Cholesky: {calculaCholesky(A)}")
    print('Ejecutando tests labo 5')
    # tests5()
    print('Ejecutando tests labo 6')
    # tests6()
    print('Ejecutando tests labo 7')
    # tests7()
    print('Ejecutando tests labo 8')
    # tests8()
    print('Pasaron todos los tests')
    
# endregion

# region Helpers TP
# Input: Triang matriz triangular (superior o inferior), b vector de terminos independientes, inferior booleano que indica si la matriz es inferior
# Output: X solucion del sistema triangular
def res_tri_mat(Triang, Y, inferior=True):
    X_cols = [] 
    Y_cols_count = Y.shape[1]
    for i in range(Y_cols_count):
        X_cols.append(res_tri(Triang, Y[:, i], inferior))
    #print(f"X_cols {X_cols}")
    X = traspuesta(np.array(X_cols))
    return X

def hermitiana(A):
    return traspuesta(A) #CORRECTO PARA DATOS REALES :D 

# Input: Lista X
# Output: Matriz diagonal con los elementos de X en la diagonal
def list_to_diag(X):
    n = len(X)
    matriz_diagonal:np.ndarray = np.zeros((n,n))
    for i in range(n):
        matriz_diagonal[i][i]=X[i]
    return matriz_diagonal

# Input: Se le pasan X e Y de un tamano mayor a cant
# Output: Se devuelven X e Y reducidas cantidad cant de perros y cantidad cant de gatos
def reducir_matrices_testeo(X, Y):
    cant = 100
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

# Input: Se ingrea una matriz A, un metodo para descomponer en QR (HouseHolder o Gram Schmidt) y una cierta tolerancia
# Output: Matrices Q y R de la descomposicion QR de A en su version reducida
def QR_reducida(A, metodo='RH', tol=1e-12):
    _, cols = A.shape
    Q, R = calculaQR(A, metodo, tol)
    # Asumimos que la cantidad de columnas es mayor a la cantidad de filas, por enunciado (1536x2000)
    return Q[:, :cols], R[:cols,:]

# Input: vector de [0,1] o [1,0] y n cantidad de columnas
# Output: Matriz Y con el vector repetido n veces como columnas
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
# endregion

# region Funciones TP
# Input: Un path
# Output: Devuelve embeddings X e Y de entrenamiento y validacion
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

# region Algoritmo 1

# Input: X matriz de embeddings, L la matriz de Cholesky y Y matriz de targets de entrenamiento
# Output: W matriz de pesos
def fullyConnectedLineal_Cholesky(X:np.ndarray, Y:np.ndarray):
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

# Input: X matriz de embeddings, L la matriz de Cholesky y Y matriz de targets de entrenamiento
# Output: W matriz de pesos
def pinvEcuacionesNormales(X, L, Y):
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

# Input: X matriz de embeddings, L la matriz de Cholesky y Y matriz de targets de entrenamiento
# Output: W matriz de pesos
def fullyConnectedLineal_SVD(X:np.ndarray, Y:np.ndarray):
    X, Y = reducir_matrices_testeo(X, Y)
    n,_ = X.shape
    U_de_x,Sigma_de_x,V_de_x = svd_reducida(X,k=n)
    return pinvSVD(U_de_x, Sigma_de_x, V_de_x,Y)

# Input: U,S,V matrices de la descomposicion SVD de X, Y matriz de targets de entrenamiento
# Output: W matriz de pesos
def pinvSVD(U:np.ndarray,S:np.ndarray,V:np.ndarray,Y:np.ndarray):
    Ur, Sr, Vr = reducirSVD(U, S, V)
    Ur_traspuesta = traspuesta(Ur)
    Sr_inversa = list_to_diag(1.0 / Sr)  
    X_inversa = productoMatricial(productoMatricial(Vr,Sr_inversa),Ur_traspuesta)
    return productoMatricial(Y,X_inversa)

# Asumimos que S tiene algun valor singular no nulo
def reducirSVD(U:np.ndarray, S:np.ndarray, V:np.ndarray, atol=1e08):
    cantidad_de_rs = 0
    n = min(S.shape[0], S.shape[1])
    for i in range(n):
        if allclose(0, S[i,i], atol):
            break
        cantidad_de_rs += 1

    return U[:, :cantidad_de_rs], S[:cantidad_de_rs, :cantidad_de_rs], V[:cantidad_de_rs, :] 

# endregion

# region Algoritmo 3

# Input: X matriz de embeddings, L la matriz de Cholesky y Y matriz de targets de entrenamiento
# Output: matriz W de pesos
def fullyConnectedLineal_QR(X:np.ndarray, Y:np.ndarray, metodo='GS'):
    #X, Y = reducir_matrices_testeo(X, Y)
    
    Q, R = QR_reducida(traspuesta(X), metodo)
    W = pinvHouseHolder(Q, R, Y)
    return W


# ! Da igual el metodo utilizado, dado a que se la pasa la Q,R ya factorizada
# Input: Q,R matrices de la descomposicion QR de Xt, Y matriz de targets de entrenamiento
# Output: W matriz de pesos
def pinvHouseHolder(Q:np.ndarray, R:np.ndarray, Y:np.ndarray):    
    V_t = res_tri_mat(R, traspuesta(Q))
    V = traspuesta(V_t)
    W = productoMatricial(Y, V)
    return W

# ! Da igual el metodo utilizado, dado a que se la pasa la Q,R ya factorizada
# Input: Q,R matrices de la descomposicion QR de Xt, Y matriz de targets de entrenamiento
# Output: W matriz de pesos
def pinvGramSchmidt(Q, R, Y):
    return pinvHouseHolder(Q, R, Y)

# endregion
# Input: Dado una matriz X, pX pseudo inversa de X y una toleracia tol
# Output: True si pX es pseudo inversa de X, False en caso contrario
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

# endregion

def main():
    # Esta seccion la utilizamos para debugging
    return 

if __name__ == "__main__":
    main()
