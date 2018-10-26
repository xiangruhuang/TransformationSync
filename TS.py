import numpy as np
from scipy.stats import special_ortho_group as so
from scipy.linalg import sqrtm
import glob
from scipy.linalg import block_diag

def __decompose__(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t

def __pack__(R, t):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T

def inverse(T):
    R, t = __decompose__(T)
    invT = np.zeros((4, 4))
    invT[:3, :3] = R.T
    invT[:3, 3] = -R.T.dot(t)
    invT[3, 3] = 1
    return invT

"""
    Find a matrix Q \in O(n) such that \|A Q - B\|_F is minimized
    equivalent to maximize trace of (Q^T A^T B)
"""
def project(A, B):
    X = A.T.dot(B)
    U, S, VT = np.linalg.svd(X)
    Q = U.dot(VT)
    return Q

"""
    Find a matrix Q \in SO(n) such that \|Q - X\|_F is minimized
    equivalent to project(I, X)
"""
def project_so(X):
    d = X.shape[0]
    assert X.shape[1] == d
    Q = project(np.eye(d), X)
    Q = Q * np.linalg.det(Q)
    return Q

def read_npys(shapeid):
    pairs = glob.glob('../rSyncPairwiseCode/matching/{}/{}_{}.npy'.format(shapeid, '*', '*'))
    ans = []
    n = 0
    for pair in pairs:
        i, j = [int(token) for token in pair.strip().split('/')[-1].split('.npy')[0].split('_')]
        Tij = np.load(pair).item()['R']
        Rij, tij = __decompose__(Tij)
        ans.append({'src':i, 'tgt':j, 'R':Rij, 't':tij, 'weight': 1.0})
        n = max(n, i+1)
        n = max(n, j+1)

    ans = np.array(ans, dtype=np.object)
    return int(n), ans

def read_npy_pair(shapeid, src, tgt):
    pairs = glob.glob('../rSyncPairwiseCode/matching/{}/{}_{}.npy'.format(shapeid, src, tgt))
    ans = []
    n = 0
    for pair in pairs:
        i, j = [int(token) for token in pair.strip().split('/')[-1].split('.npy')[0].split('_')]
        Tij = np.load(pair).item()['R']
        Rij, tij = __decompose__(Tij)
        ans.append({'src':i, 'tgt':j, 'R':Rij, 't':tij, 'weight': 1.0})
        n = max(n, i+1)
        n = max(n, j+1)

    ans = np.array(ans, dtype=np.object)
    return int(n), ans


def read_npy(shapeid):
    infile = glob.glob('../rSyncPairwiseCode/results/{}.npy'.format(shapeid))
    fin = np.load(infile[0])
    edges = fin.item()['edges']
    for edge in edges:
        edge['weight'] = 1.0
    print(fin.item())
    Pose = fin.item()['Pose']
    n = Pose.shape[0]

    return n, edges, Pose

def generate_synthetic(n, sigma):
    T = np.zeros((n, 4, 4))
    X = so.rvs(dim=3, size=n)
    T[:, :3, :3] = X
    # u, sigma, v = np.linalg.svd(T[0])
    T[:, :3, 3] = np.random.randn(n, 3)
    T[0, :3, 3] = 0.0
    T[:, 3, 3] = 1
    edges = []
    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            Tij = T[j].dot(inverse(T[i]))
            Rij, tij = __decompose__(Tij)
            Rij = Rij + np.random.randn(3, 3) * sigma
            Rij = project_so(Rij)
            tij = tij + np.random.randn(3) * sigma
            Tij = __pack__(Rij, tij)
            edge = {'src':i,
                    'tgt':j,
                    'R': Rij,
                    't': tij,
                    'weight': 1.0}
            edges.append(edge)
    edges = np.array(edges)
    return n, edges, T

"""
    Construct Normalized Adjacency Matrix 
    Anorm(i, j) = {
        wij Rij.T / sqrt(di dj),  if (i, j) is an edge
        0, o.w. 
    }
"""
def __normalized_adjacency__(n, edges):
    A = np.zeros((3*n, 3*n))
    deg = np.zeros(n)
    Adj = np.zeros((n, n))
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        weight = edge['weight']
        assert i > j
        A[(i*3):(i+1)*3, (j*3):(j+1)*3] = weight*Rij.T
        A[(j*3):(j+1)*3, (i*3):(i+1)*3] = weight*Rij
        deg[i] += weight
        deg[j] += weight

    Dinv = np.kron(np.diag(1.0/deg), np.eye(3))

    Anorm = sqrtm(Dinv).dot(A).dot(sqrtm(Dinv))
    return Anorm, deg

""" Estimate Absolute Rotation from Relative Rotation
    n: number of vertices
    edges: array of np.object, each contains items [i, j, Tij, weight]
"""
def Spectral(n, edges):
    Anorm, deg = __normalized_adjacency__(n, edges)
    Dinv = np.kron(np.diag(1.0/deg), np.eye(3))

    lamb, V = np.linalg.eigh(Anorm)
    eigengap = lamb[-3] - lamb[-4]
    V = V[:, -3:]
    dsqrt = np.sqrt(deg.sum())
    V = sqrtm(Dinv).dot(V) * dsqrt

    R = []
    for i in range(n):
        Ri = V[i*3:(i+1)*3, :]
        Ri = project_so(Ri)
        R.append(Ri)

    R = np.array(R)

    return R, eigengap

"""
    Solve min_{ti, tj} \sum_{i, j} wij \| Rj tij + tj - ti \|^2
    Gradient Descent (only for debug)
"""
def __LeastSquaresGD__(n, edges, R):
    t = np.zeros((n, 3))
    lr = 1e-5
    for itr in range(10000):
        grad = np.zeros((n, 3))
        loss = 0.0
        for edge in edges:
            i = edge['i']
            j = edge['j']
            Rij = edge['Tij'][:3, :3]
            tij = edge['Tij'][:3, 3]
            weight = edge['weight']
            loss += weight * np.linalg.norm(Rij.dot(t[i]) + tij - t[j], 2) ** 2
            grad[i, :] += 2.0*weight*(t[i] + Rij.T.dot(tij) - Rij.T.dot(t[j]))
            grad[j, :] += 2.0*weight*(t[j] - Rij.T.dot(t[i]) - tij)
        t = t - grad * lr
        if itr % 100 == 0:
            print('iter=%d, loss=%f' % (itr, loss))
    return t

"""
    Solve min_{ti, tj} \sum_{i, j} wij \| Rj tij + tj - ti \|^2
    By solving linear equation At = b.
"""
def LeastSquares(n, edges):
    Anorm, deg = __normalized_adjacency__(n, edges)
    Lnorm = np.eye(n*3) - Anorm
    #_, nullL = np.eigh(Anorm)
    #nullL = nullL(:, -3:)
    D = np.kron(np.diag(deg), np.eye(3))
    L = sqrtm(D).dot(Lnorm).dot(sqrtm(D))
    #Lnorm = np.eye(3)
    #Lnorm = np.eye(3)

    b = np.zeros(n*3)
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        Rij = edge['R']
        tij = edge['t']
        weight = edge['weight']
        
        #L[i*3:(i+1)*3, j*3:(j+1)*3] -= weight * Rij.T
        #L[j*3:(j+1)*3, i*3:(i+1)*3] -= weight * Rij
        #L[i*3:(i+1)*3, i*3:(i+1)*3] += weight * np.eye(3)
        #L[j*3:(j+1)*3, j*3:(j+1)*3] += weight * np.eye(3)
        b[i*3:(i+1)*3] += weight*(-Rij.T).dot(tij)
        b[j*3:(j+1)*3] += weight*tij

    t = np.linalg.lstsq(L, b)[0]
    #print(L.shape, t.shape)
    #print('Loss=%f' % np.linalg.norm(L.dot(t) - b, 2))
    return t

def find(x, f):
    if f[x] == f[f[x]]:
        

def connected(n, edges):
    f = np.zeros(n)
    for i in range(n):
        f[i] = i
    for edge in edges:
        i = edge['src']
        j = edge['tgt']
        

def TruncatedRotSync(n, edges, eps0=-1, decay=0.999, Tstar=None, max_iter=10000):
    itr = 0
    while itr < max_iter:
        R, eigengap = Spectral(n, edges)
        t = LeastSquares(n, edges)
        
        # print('Translation Loss=%f' % tloss)
        # print('Diff to Ground Truth=%f' % np.linalg.norm(np.reshape(Tstar[:, :3, 3], n*3)-t, 2))
        err_sum = 0.0
        err_max = 0.0
        if eps0 < -0.5:
            for edge in edges:
                i = edge['src']
                j = edge['tgt']
                Rij = edge['R']
                weight = edge['weight']
                err_e = np.linalg.norm(R[j].dot(R[i].T) - Rij, 2)
                if eps0 < err_e:
                    eps0 = err_e
            print('setting threshold to %f' % eps0)

        numedges = 0
        max_existing_err = 0.0
        cover = np.zeros(n)
        for edge in edges:
            i = edge['src']
            j = edge['tgt']
            assert i >= j
            Rij = edge['R']
            weight = edge['weight']
            err = np.linalg.norm(R[j].dot(R[i].T) - Rij, 2) * weight
            err_sum += err
            err_max = max(err_max, err)
            if weight < 1e-10:
                continue
            if err > eps0 - 1e-12:
                edge['weight'] = 0.0
            elif err > max_existing_err - 1e-12:
                max_existing_err = err
            
            cover[i] += edge['weight']
            cover[j] += edge['weight']
            numedges += edge['weight']
        print('iter=%d, max(err)=%f, avg(err)=%f, eigengap=%f, #edges=%d, min_deg=%f, eps0=%f' % (itr, err_max, err_sum/len(edges), eigengap, numedges, min(cover), eps0))
        while (itr < max_iter) and (eps0 > max_existing_err):
            eps0 = eps0 * decay
            itr += 1
        if err_sum <= 1e-1:
            break
    return R
        

""" Ground Truth """
Tstar = None
# n, edges, Tstar = generate_synthetic(30, 0.001)

# Rstar = Tstar[:, :3, :3]
# Rdiag = block_diag(*Rstar)
# Rvec = np.reshape(Rstar, [3*n, 3])

# n, edges = read_npys('00021')
n, edges, Tstar = read_npy('00021')

R = TruncatedRotSync(n, edges, Tstar=Tstar)

