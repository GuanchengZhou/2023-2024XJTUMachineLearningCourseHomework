import numpy as np

def l2_distance(a):
    """
    Calculate L2 Distance
    :param a:
    :return:
    """

class kmeans_class:
    def __init__(self, k=3, distance_type='l2'):
        self.k = k
        self.centers = None
    def train(self, a, lim_iter=100):
        n = len(a)

        # init
        center_ids = np.random.choice(n, self.k)
        centers = np.array([a[i] for i in center_ids])
        print(centers)
        run_flag = True
        iter = 0

        while run_flag and (iter < lim_iter):
            print(centers)
            iter += 1
            delta = (np.repeat(np.expand_dims(a, axis=1), self.k, axis=1) - centers)
            dis = np.einsum('ijk,ijk->ij', delta, delta)
            _class = np.argmin(dis, axis=1)
            print('class', _class)
            new_centers = []
            for i in range(self.k):
                new_centers.append(np.mean(a[_class == i], axis=0))
            new_centers = np.array(new_centers)
            if (new_centers == centers).all():
                run_flag = False
            else:
                run_flag = True
            centers = np.array(new_centers)

        return centers