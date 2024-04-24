import numpy as np
import matplotlib.pyplot as plt

def k_means(a, k=3, lim_iter=100):
    n = len(a)

    # init
    center_ids = np.random.choice(n, k)
    centers = np.array([a[i] for i in center_ids])
    print(centers)
    run_flag = True
    iter = 0

    while run_flag and (iter<lim_iter):
        print(centers)
        iter += 1
        delta = (np.repeat(np.expand_dims(a, axis=1), k, axis=1)-centers)
        print('1', np.repeat(np.expand_dims(a, axis=1), k, axis=1).shape, '2', centers.shape)
        dis = np.einsum('ijk,ijk->ij', delta, delta)
        print('dis', dis.shape)
        _class = np.argmin(dis, axis=1)
        print('class', _class)
        new_centers = []
        for i in range(k):
            new_centers.append(np.mean(a[_class==i], axis=0))
        new_centers = np.array(new_centers)
        if (new_centers == centers).all():
            run_flag = False
        else:
            run_flag = True
        centers = np.array(new_centers)
    
    return centers

def predict_kmeans(X, centers):
    y = []
    for x in X:
        dis = []
        for center in centers:
            dis.append((np.sum(x-center)**2))
        y.append(np.argmin(dis))
    return np.array(y)

if __name__=='__main__':
    # a = np.random.randint(0, 100, (10, 2))
    a = np.random.normal(0, 1, size=(10,2))
    a = np.append(a, np.random.normal(2, 1, size=(10,2)), axis=0)

    # print(a)
    centers = k_means(a, k=2, lim_iter=100)
    print(centers)

    y = predict_kmeans(a, centers)

    # print(y)

    cs = ['r', 'g', 'b']

    for i in range(len(centers)):
        plt.scatter(a[y==i, 0], a[y==i,1], c=cs[i])
        plt.scatter(centers[i, 0], centers[i, 1], c=cs[i], marker='+')

    # plt.scatter(a[:, 0], a[:,1], c='b')
    # plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='+')
    plt.show()

