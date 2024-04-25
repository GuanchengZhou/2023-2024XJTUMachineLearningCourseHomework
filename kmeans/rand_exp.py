from kmeans_class import *
from read_iris import *
import matplotlib.pyplot as plt

def get_random_data(mu=(0,0), sigma=(1,1), n=10, label=1):
    X = [(np.random.normal(mu[0], sigma[0]), np.random.normal(mu[1], sigma[1])) for i in range(n)]
    Y = [label for i in range(n)]
    return np.array(X), np.array(Y)

if __name__=='__main__':
    k = 4
    n = 100
    scale = 2.5
    locs = [(0,0), (scale,scale), (0,scale), (scale,0)]
    # locs = [(0,0), (0,scale), (scale,0), (scale*5, scale*5)]
    cs = ['r', 'g', 'b', 'y', 'c', 'deeppink', 'lightcoral', 'peru']

    train_x, train_y = get_random_data(mu=locs[0], n=n, label=0)
    test_x, test_y = get_random_data(mu=locs[0], n=int(n*0.25), label=0)
    # train_x = np.append(train_x, np.array([[10,10]]), axis=0)
    for i in range(k-1):
        x, y = get_random_data(mu=locs[i+1], n=n, label=i+1)
        train_x = np.append(train_x, x, axis=0)
        train_y = np.append(train_y, y, axis=0)
        x, y = get_random_data(mu=locs[i + 1], n=int(n*0.25), label=i+1)
        test_x = np.append(test_x, x, axis=0)
        test_y = np.append(test_y, y, axis=0)

    model = kmeans(k=k, distance_type='l2')

    for i in range(10):
        model.train(train_x, lim_iter=100)

        y = model.predict_kmeans(train_x)
        y = model.predict_kmeans(test_x)

        print('l2')
        evaluate_external(test_y, y, k=k)



    # for i in range(len(model.centers)):
    #     if i==0:
    #         plt.scatter(train_x[y == i, 0], train_x[y == i, 1], c=cs[i], s=20, label='Cluster')
    #         plt.scatter(model.centers[i, 0], model.centers[i, 1], c=cs[i+k], marker='+', s=35, label='Center of Cluster')
    #         plt.scatter(locs[i][0], locs[i][1], c=cs[i+k], marker='^', s=20, label='Ground Truth Center')
    #     else:
    #         plt.scatter(train_x[y == i, 0], train_x[y == i, 1], c=cs[i], s=20)
    #         plt.scatter(model.centers[i, 0], model.centers[i, 1], c=cs[i], marker='+', s=35)
    #         plt.scatter(locs[i][0], locs[i][1], c=cs[i + k], marker='^', s=20)
    #
    # plt.legend()
    # plt.show()
    # plt.cla()

    model = kmeans(k=k, distance_type='l1')

    for i in range(10):
        model.train(train_x, lim_iter=100)

        y = model.predict_kmeans(train_x)
        y = model.predict_kmeans(test_x)

        print('l1')
        evaluate_external(test_y, y, k=k)

    # for i in range(len(model.centers)):
    #     if i == 0:
    #         plt.scatter(train_x[y == i, 0], train_x[y == i, 1], c=cs[i], s=20, label='Cluster')
    #         plt.scatter(model.centers[i, 0], model.centers[i, 1], c=cs[i+k], marker='+', s=35, label='Center of Cluster')
    #         plt.scatter(locs[i][0], locs[i][1], c=cs[i + k], marker='^', s=20, label='Ground Truth Center')
    #     else:
    #         plt.scatter(train_x[y == i, 0], train_x[y == i, 1], c=cs[i], s=20)
    #         plt.scatter(model.centers[i, 0], model.centers[i, 1], c=cs[i], marker='+', s=35)
    #         plt.scatter(locs[i][0], locs[i][1], c=cs[i + k], marker='^', s=20)
    #
    # plt.legend()
    # plt.show()