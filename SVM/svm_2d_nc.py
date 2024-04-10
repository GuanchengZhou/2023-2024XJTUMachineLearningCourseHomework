import numpy as np
from sklearn.svm import SVC
from get_mnist import *
from sklearn.preprocessing import StandardScaler
from get_rand_data import *
import matplotlib.pyplot as plt

class ovr_svc:
    def __init__(self, n_cls):
        """
        Initial the ovr SVC
        :param n_cls: the number of classes
        """
        self.n_cls = n_cls
        self.models = [SVC(C=1.0, kernel='linear', probability=True) for i in range(n_cls)]
        self.intercept_ = []
        self.coef_ = []
    def fit(self, X, Y):
        """
        Train the SVC
        :param X: Data
        :param Y: Label
        """
        for i in range(self.n_cls):
            _Y = Y.copy()
            _Y[Y==i+1] = 0
            _Y[Y!=i+1] = -1
            self.models[i].fit(X,_Y)
            self.intercept_.append(self.models[i].intercept_[0])
            self.coef_.append(self.models[i].coef_[0])
    def predict(self, X):
        """
        Predict the label of data
        :param X: Data
        :return: Predicted label
        """
        result = []
        for i in range(self.n_cls):
            # print(self.models[i].predict_proba(X))
            result.append(self.models[i].predict_proba(X)[:,-1])
        result = np.array(result)
        result = np.argmax(result, axis=0) + 1
        return result

if __name__=='__main__':
    # Read Data

    colors = ['y', 'c', 'indigo', 'deeppink', 'lightcoral', 'black', 'peru']

    n = 4

    train_x1, train_y1 = get_random_data(label=1, n=100)
    train_x2, train_y2 = get_random_data((3,0), (1,1), label=2, n=100)
    train_x3, train_y3 = get_random_data((1.5,1.5*np.sqrt(3)), (1,1), label=3, n=100)
    train_x4, train_y4 = get_random_data((1.5, -1.5 * np.sqrt(3)), (1, 1), label=4, n=100)
    train_x = np.append(train_x1, train_x2, axis=0)
    train_x = np.append(train_x, train_x3, axis=0)
    train_x = np.append(train_x, train_x4, axis=0)
    train_y = np.append(train_y1, train_y2, axis=0)
    train_y = np.append(train_y, train_y3, axis=0)
    train_y = np.append(train_y, train_y4, axis=0)

    test_x1, test_y1 = get_random_data(label=1, n=50)
    test_x2, test_y2 = get_random_data((3, 0), (1, 1), label=2, n=50)
    test_x3, test_y3 = get_random_data((1.5, 1.5*np.sqrt(3)), (1, 1), label=3, n=50)
    test_x4, test_y4 = get_random_data((1.5, -1.5 * np.sqrt(3)), (1, 1), label=4, n=50)
    test_x = np.append(test_x1, test_x2, axis=0)
    test_x = np.append(test_x, test_x3, axis=0)
    test_x = np.append(test_x, test_x4, axis=0)
    test_y = np.append(test_y1, test_y2, axis=0)
    test_y = np.append(test_y, test_y3, axis=0)
    test_y = np.append(test_y, test_y4, axis=0)

    # print(train_x.shape, train_y.shape)

    model = SVC(C=1.0, kernel='linear', decision_function_shape='ovo')
    model2 = ovr_svc(n)

    model.fit(train_x, train_y)
    model2.fit(train_x, train_y)

    model2.predict(train_x)

    print(len(model2.intercept_))


    pred_y = model.predict(test_x)
    print(pred_y)
    pr = np.sum(pred_y==test_y) / len(pred_y) * 100
    print('test', pr,'%')

    pred_y = model.predict(train_x)
    pr = np.sum(pred_y == train_y) / len(pred_y) * 100
    print('train', pr, '%')

    for i in range(int(n*(n-1)/2)):
        p1, p2 = model.coef_[i]
        b = model.intercept_[i]
        plt.plot(train_x[:,0], -p1/p2*train_x[:,0]-b/p2, c=colors[i], label='Predict Line {}'.format(i))

    plt.scatter(train_x1[:,0], train_x1[:,1], marker='.', c='b')
    plt.scatter(train_x2[:,0], train_x2[:,1], marker='+', c='r')
    plt.scatter(train_x3[:, 0], train_x3[:, 1], marker='*', c='g')
    plt.scatter(train_x4[:, 0], train_x4[:, 1], marker='x', c='black')

    print(train_x.shape)

    plt.legend()

    plt.xlim(np.min(train_x[:, 0]), np.max(train_x[:, 0]))
    plt.ylim(np.min(train_x[:, 1]), np.max(train_x[:, 1]))

    plt.show()

    plt.cla()

    pred_y = model2.predict(test_x)
    pr = np.sum(pred_y == test_y) / len(pred_y) * 100
    print('test', pr, '%')

    pred_y = model2.predict(train_x)
    pr = np.sum(pred_y == train_y) / len(pred_y) * 100
    print('train', pr, '%')

    for i in range(n):
        p1, p2 = model2.coef_[i]
        b = model2.intercept_[i]
        plt.plot(train_x[:, 0], -p1 / p2 * train_x[:, 0] - b / p2, c=colors[i], label='Predict Line {}'.format(i))

    plt.scatter(train_x1[:, 0], train_x1[:, 1], marker='.', c='b')
    plt.scatter(train_x2[:, 0], train_x2[:, 1], marker='+', c='r')
    plt.scatter(train_x3[:, 0], train_x3[:, 1], marker='*', c='g')
    plt.scatter(train_x4[:, 0], train_x4[:, 1], marker='x', c='black')

    print(train_x.shape)

    plt.legend()

    plt.xlim(np.min(train_x[:, 0]), np.max(train_x[:, 0]))
    plt.ylim(np.min(train_x[:, 1]), np.max(train_x[:, 1]))

    plt.show()

    # scaler = StandardScaler()
    # scaler.fit(train_x)

    # train_x_std = scaler.transform(train_x)

    # model = ovo_svm(10, 10)


