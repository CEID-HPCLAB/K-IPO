import numpy as np

from scipy.sparse import csc_matrix, lil_matrix, diags

from sklearn.metrics.pairwise import pairwise_distances


"""
Laplacian Score

References:
    Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R. P.,
    Tang, J., Liu, H. (2018). Feature Selection: A Data Perspective.
    ACM Computing Surveys (CSUR), 50(6), Article 94.

Source:
    Adapted from the scikit-feature library: https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/lap_score.py
"""


def laplacian_score(X, y):
    kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
    
    W = construct_W(X, ** kwargs); D = np.array(W.sum(axis = 1)); L = W
    
    tmp = np.dot(np.transpose(D), X); D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
   
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    
    D_prime[D_prime < 1e-12] = 10000
    
    lap_score = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]

    return np.transpose(lap_score)


def construct_W(X, ** kwargs):
    if 'metric' not in kwargs.keys():
        kwargs['metric'] = 'cosine'

    if 'neighbor_mode' not in kwargs.keys():
        kwargs['neighbor_mode'] = 'knn'
    if kwargs['neighbor_mode'] == 'knn' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'y' not in kwargs.keys():
        print('Warning: label is required in the supervised neighborMode!!!')
        exit(0)

    if 'weight_mode' not in kwargs.keys():
        kwargs['weight_mode'] = 'binary'
    if kwargs['weight_mode'] == 'heat_kernel':
        if kwargs['metric'] != 'euclidean':
            kwargs['metric'] = 'euclidean'
        if 't' not in kwargs.keys():
            kwargs['t'] = 1
    elif kwargs['weight_mode'] == 'cosine':
        if kwargs['metric'] != 'cosine':
            kwargs['metric'] = 'cosine'

    if 'fisher_score' not in kwargs.keys():
        kwargs['fisher_score'] = False
    if 'reliefF' not in kwargs.keys():
        kwargs['reliefF'] = False

    n_samples = np.shape(X)[0]

    if kwargs['neighbor_mode'] == 'knn':
        k = kwargs['k']
        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                D = pairwise_distances(X); D **= 2
              
                dump = np.sort(D, axis = 1); idx = np.argsort(D, axis = 1)
              
                idx_new = idx[:, 0:k + 1]
                
                G = np.zeros((n_samples* (k + 1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1); G[:, 1] = np.ravel(idx_new, order = 'F'); G[:, 2] = 1
               
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
                bigger = np.transpose(W) > W
                
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

            elif kwargs['metric'] == 'cosine':
                X_normalized = np.power(np.sum(X*X, axis = 1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
                
                D_cosine = np.dot(X, np.transpose(X))
                dump = np.sort(-D_cosine, axis = 1)
                
                idx = np.argsort(-D_cosine, axis = 1); idx_new = idx[:, 0:k + 1]
                
                G = np.zeros((n_samples * (k + 1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1); G[:, 1] = np.ravel(idx_new, order = 'F'); G[:, 2] = 1

                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
                bigger = np.transpose(W) > W
                
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            t = kwargs['t']

            D = pairwise_distances(X); D **= 2
            
            dump = np.sort(D, axis = 1); dump_new = dump[:, 0:k + 1]
            idx = np.argsort(D, axis = 1); idx_new = idx[:, 0:k + 1]
            
            dump_heat_kernel = np.exp(-dump_new / (2 * t * t))
            
            G = np.zeros((n_samples * (k + 1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1); G[:, 1] = np.ravel(idx_new, order = 'F'); G[:, 2] = np.ravel(dump_heat_kernel, order = 'F')
            
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
            bigger = np.transpose(W) > W
            
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            
            return W

        elif kwargs['weight_mode'] == 'cosine':
            X_normalized = np.power(np.sum(X*X, axis = 1), 0.5)
            
            for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
           
            D_cosine = np.dot(X, np.transpose(X))
          
            dump = np.sort(-D_cosine, axis = 1); dump_new = -dump[:, 0:k + 1]
            idx = np.argsort(-D_cosine, axis = 1); idx_new = idx[:, 0:k + 1]
            
            G = np.zeros((n_samples * (k + 1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1); G[:, 1] = np.ravel(idx_new, order = 'F'); G[:, 2] = np.ravel(dump_new, order = 'F')
          
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
            bigger = np.transpose(W) > W
            
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            
            return W

    elif kwargs['neighbor_mode'] == 'supervised':
        k = kwargs['k']; y = kwargs['y']
        label = np.unique(y); n_classes = np.unique(y).size
        
        if kwargs['fisher_score'] is True:
            W = lil_matrix((n_samples, n_samples))
            for i in range(n_classes):
                class_idx = (y == label[i])
                # class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
                # class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
                class_idx = np.asarray(class_idx)
                class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
                W[class_idx_all] = 1.0 / np.sum(np.sum(class_idx))
            
            return W

        if kwargs['reliefF'] is True:
            G = np.zeros((n_samples * (k + 1), 3)); id_now = 0
            
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                
                D = pairwise_distances(X[class_idx, :]); D **= 2
                
                idx = np.argsort(D, axis = 1); idx_new = idx[:, 0:k + 1]
                n_smp_class = (class_idx[idx_new[:]]).size
                
                if len(class_idx) <= k:
                    k = len(class_idx) - 1
                
                G[id_now:n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                G[id_now:n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order = 'F')
                G[id_now:n_smp_class + id_now, 2] = 1.0 / k
                
                id_now += n_smp_class
            
            W1 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
            
            for i in range(n_samples):
                W1[i, i] = 1

            G = np.zeros((n_samples * k * (n_classes - 1), 3)); id_now = 0
            
            for i in range(n_classes):
                class_idx1 = np.column_stack(np.where(y == label[i]))[:, 0]
                X1 = X[class_idx1, :]
                
                for j in range(n_classes):
                    if label[j] != label[i]:
                        class_idx2 = np.column_stack(np.where(y == label[j]))[:, 0]
                        X2 = X[class_idx2, :]
                        
                        D = pairwise_distances(X1, X2)
                        idx = np.argsort(D, axis = 1); idx_new = idx[:, 0:k]
                        
                        n_smp_class = len(class_idx1) * k
                        
                        G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx1, (k, 1)).reshape(-1)
                        G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx2[idx_new[:]], order = 'F')
                        G[id_now:n_smp_class+id_now, 2] = -1.0/((n_classes - 1) * k)
                        id_now += n_smp_class
            
            W2 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
            bigger = np.transpose(W2) > W2
            
            W2 = W2 - W2.multiply(bigger) + np.transpose(W2).multiply(bigger)
            W = W1 + W2
            
            return W

        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                G = np.zeros((n_samples * (k + 1), 3)); id_now = 0
                
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                   
                    D = pairwise_distances(X[class_idx, :]); D **= 2
                
                    idx = np.argsort(D, axis = 1); idx_new = idx[:, 0:k + 1]
                    
                    n_smp_class = len(class_idx) * (k + 1)
                    
                    G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                    G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order = 'F')
                    G[id_now:n_smp_class+id_now, 2] = 1
                    
                    id_now += n_smp_class

                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
                bigger = np.transpose(W) > W
                
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                
                return W

            if kwargs['metric'] == 'cosine':
                X_normalized = np.power(np.sum(X*X, axis = 1), 0.5)
                
                for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
                
                G = np.zeros((n_samples* (k + 1), 3)); id_now = 0
                
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]

                    D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                    
                    idx = np.argsort(-D_cosine, axis = 1); idx_new = idx[:, 0:k + 1]
                    
                    n_smp_class = len(class_idx) * (k + 1)
                    
                    G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                    G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order = 'F')
                    G[id_now:n_smp_class+id_now, 2] = 1
                    
                    id_now += n_smp_class

                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
                bigger = np.transpose(W) > W
                
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            G = np.zeros((n_samples* (k + 1), 3))
            
            id_now = 0
            
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]

                D = pairwise_distances(X[class_idx, :]); D **= 2
                
                dump = np.sort(D, axis = 1);  dump_new = dump[:, 0:k + 1]
                idx = np.argsort(D, axis = 1); idx_new = idx[:, 0:k + 1]
               
                t = kwargs['t']
                dump_heat_kernel = np.exp(-dump_new/(2 * t * t))
                
                n_smp_class = len(class_idx) * (k + 1)
                
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order = 'F')
                G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_heat_kernel, order = 'F')
                
                id_now += n_smp_class

            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
            bigger = np.transpose(W) > W
            
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            
            return W

        elif kwargs['weight_mode'] == 'cosine':
            X_normalized = np.power(np.sum(X*X, axis = 1), 0.5)
            
            for i in range(n_samples):
                X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
            
            G = np.zeros((n_samples* (k + 1), 3)); id_now = 0
            
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                
                D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                
                dump = np.sort(-D_cosine, axis = 1); dump_new = -dump[:, 0:k + 1]
                idx = np.argsort(-D_cosine, axis = 1); idx_new = idx[:, 0:k + 1]
               
                n_smp_class = len(class_idx) * (k + 1)
                
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order = 'F')
                G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_new, order = 'F')
                
                id_now += n_smp_class
           
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape = (n_samples, n_samples))
            bigger = np.transpose(W) > W
            
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            
            return W