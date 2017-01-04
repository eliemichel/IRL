import numpy as np
from numpy.linalg import inv

"""
Functions ported from MATLAB code provided or written during TPs
"""

def build_similarity_graph(X, graph_param):
    graph_type = graph_param['graph_type']
    graph_thresh = graph_param['graph_thresh']  # the number of neighbours for the graph
    sigma2 = graph_param['sigma2']  # exponential_euclidean's sigma^2

    Xsq = X.dot(X.T)
    D = np.diag(Xsq).reshape(1, -1)
    Y = D + D.T
    similarities = np.exp(-np.sqrt(Y - 2 * Xsq) / (2 * sigma2));

    n = X.shape[0]

    W = np.zeros((n,n))

    if graph_type == 'knn':
        I_n = np.eye(n)
        mask = np.zeros((n,n), bool)  # values we keep in similarities
        for i in range(int(graph_thresh)):
            index = np.argmax(similarities * (~mask), axis=1)
            mask |= (I_n[index, :] == 1)
        mask |= mask.T  # symetrize mask (graph is undirected for OR knn)
        W = similarities * mask


    elif graph_type == 'eps':
        W = similarities * (similarities >= graph_thresh)

    else:
        raise ValueError('build_similarity_graph: not a valid graph type')

    return W, similarities


def build_laplacian_regularized(X, graph_param, laplacian_param):
    W, _ = build_similarity_graph(X, graph_param)

    laplacian_normalization = laplacian_param['normalization']
    laplacian_regularization = laplacian_param['regularization']

    d = np.sum(W, axis=0)
    I = np.eye(*W.shape)

    if laplacian_normalization == 'unn':
        D = np.diag(d)
        L = D - W
    elif laplacian_normalization == 'sym':
        sqrt_inv_D = np.diag(d ** (-0.5))
        L = I - sqrt_inv_D.dot(W).dot(sqrt_inv_D)
    elif laplacian_normalization == 'rw':
        inv_D = np.diag(d ** (-1))
        L = I - inv_D.dot(W)
    else:
        error('unkown normalization mode')

    Q = L + laplacian_regularization * I;

    return Q


def hard_hfs(X, Y, graph_param, laplacian_param):
    num_samples = X.shape[0]
    num_classes = Y.max() + 1

    l_idx = np.where(Y>=0)[0]
    u_idx = np.where(Y<0)[0]
    l = l_idx.shape[0]

    Ic = np.eye(num_classes)
    y = Ic[Y[l_idx], :]

    Q = build_laplacian_regularized(X, graph_param, laplacian_param)
    W, _ = build_similarity_graph(X, graph_param)

    Q_uu = Q[u_idx, :][:, u_idx]
    W_ul = W[u_idx, :][:, l_idx]

    f_l = y
    f_u = inv(Q_uu).dot(W_ul.dot(f_l))

    f = np.ndarray((num_samples, num_classes))
    f[u_idx] = f_u
    f[l_idx] = f_l
    labels = np.argmax(f, axis=1)
    return labels
