import numpy as np
from sklearn.metrics import rand_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
import itertools

def l2_distance(a, centers):
    """
    Calculate L2 Distance
    :param a: N*C
    :param centers: K*C
    :return: N*K
    """
    k = len(centers)
    delta = (np.repeat(np.expand_dims(a, axis=1), k, axis=1) - centers) # N*k*C
    # dis = np.einsum('ijk,ijk->ij', delta, delta)
    dis = np.sum(np.abs(delta)**2, axis=2)
    return dis

def l2_update(a, _class, k):
    new_centers = []
    for i in range(k):
        new_centers.append(np.mean(a[_class == i], axis=0))
    return np.array(new_centers)


def l1_distance(a, centers):
    """
    Calculate L1 Distance
    :param a: N*C
    :param centers: K*C
    :return: N*K
    """
    k = len(centers)
    delta = (np.repeat(np.expand_dims(a, axis=1), k, axis=1) - centers)  # N*k*C
    dis = np.sum(np.abs(delta), axis=2)
    return dis

def l1_update(a, _class, k):
    new_centers = []
    for i in range(k):
        new_centers.append(np.median(a[_class == i], axis=0))
    return np.array(new_centers)

def linfty_distance(a, centers):
    k = len(centers)
    k = len(centers)
    delta = (np.repeat(np.expand_dims(a, axis=1), k, axis=1) - centers)  # N*k*C
    dis = np.max(np.abs(delta), axis=2)
    return dis


class kmeans:
    def __init__(self, k=3, distance_type='l2'):
        self.k = k
        self.centers = None
        self.distance_func = None
        self.update_func = None
        if distance_type=='l2':
            self.distance_func = l2_distance
            self.update_func = l2_update
        elif distance_type=='l1':
            self.distance_func = l1_distance
            self.update_func = l1_update

    def train(self, a, lim_iter=100):
        n = len(a)

        # init
        center_ids = np.random.choice(n, self.k)
        centers = np.array([a[i] for i in center_ids])
        # print(centers)
        run_flag = True
        iter = 0

        while run_flag and (iter < lim_iter):
            # print(centers)
            iter += 1
            # delta = (np.repeat(np.expand_dims(a, axis=1), self.k, axis=1) - centers)
            # dis = np.einsum('ijk,ijk->ij', delta, delta)
            dis = self.distance_func(a, centers)
            _class = np.argmin(dis, axis=1)
            # print('class', _class)
            # new_centers = []
            # for i in range(self.k):
            #     new_centers.append(np.mean(a[_class == i], axis=0))
            # new_centers = np.array(new_centers)
            new_centers = self.update_func(a, _class, self.k)
            if (new_centers == centers).all():
                run_flag = False
            else:
                run_flag = True
            centers = np.array(new_centers)
        self.centers = centers
        return centers

    def predict_kmeans(self, X):
        assert self.centers is not None, 'Please train k-means'
        # y = []
        # for x in X:
        #     dis = []
        #     for center in self.centers:
        #         dis.append((np.sum(x - center) ** 2))
        #     y.append(np.argmin(dis))
        dis = self.distance_func(X, self.centers)
        y = np.argmin(dis, axis=1)
        return y

def evaluate_external(Y_pred, Y_gt, k=None):
    RI = rand_score(Y_gt, Y_pred)
    ARI = adjusted_rand_score(Y_gt, Y_pred)
    h, c, v = homogeneity_score(Y_gt, Y_pred), completeness_score(Y_gt, Y_pred), v_measure_score(Y_gt, Y_pred)

    print('Evaluation')
    print(' RI', RI, 'ARI', ARI)
    print(' H', h, 'C', c, 'V', v)


    hashs = list(range(k))
    acc = 0
    for perm in itertools.permutations(hashs, k):
        _Y = np.zeros_like(Y_pred)
        for i in range(k):
            _Y[Y_pred==i] = perm[i]
        # print(_Y)
        acc = max(np.sum(_Y==Y_gt), acc)
    print(' Acc', acc/len(Y_pred)*100, '%')

    print(' {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f}\\%'.format(ARI, h, c, v, acc/len(Y_pred)*100))

if __name__=='__main__':
    import matplotlib.pyplot as plt
    model = kmeans(k=2, distance_type='linf')

    a = np.random.normal(0, 1, size=(10, 2))
    a = np.append(a, np.random.normal(2, 1, size=(10,2)), axis=0)

    model.train(a, lim_iter=100)

    y = model.predict_kmeans(a)

    cs = ['r', 'g', 'b']

    for i in range(len(model.centers)):
        plt.scatter(a[y == i, 0], a[y == i, 1], c=cs[i])
        plt.scatter(model.centers[i, 0], model.centers[i, 1], c=cs[i], marker='+')

    plt.show()