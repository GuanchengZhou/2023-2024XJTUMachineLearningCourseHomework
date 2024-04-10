from sklearn.svm import SVC
from get_mnist import *
from sklearn.preprocessing import StandardScaler
from get_rand_data import *
import matplotlib.pyplot as plt

if __name__=='__main__':
    # Read Data

    # dataset = mnist_loader(.8)
    # train_x, train_y = dataset.get_train_data()
    # test_x, test_y = dataset.get_test_data()
    # train_x = train_x[:1000]
    # train_y = train_y[:1000]
    # test_x = test_x[:100]
    # test_y = test_y[:100]

    train_x1, train_y1 = get_random_data(label=1, n=100)
    train_x2, train_y2 = get_random_data((2,2), (1,1), label=2, n=100)
    train_x = np.append(train_x1, train_x2, axis=0)
    train_y = np.append(train_y1, train_y2, axis=0)

    print(train_x.shape, train_y.shape)

    model = SVC(C=1.0, kernel='linear')
    model.fit(train_x, train_y)

    # print(model.coef_)
    # print(model.intercept_)

    # pred_y = model.predict(test_x)
    # print(pred_y.shape)
    # print(pred_y)
    # pr = np.sum(pred_y==test_y)
    # print(pr,'%')

    p1, p2 = model.coef_[0]
    b = model.intercept_[0]

    plt.plot(train_x[:,0], -p1/p2*train_x[:,0]-b/p2, c='y', label='Predict Line')
    plt.plot(train_x[:,0], 2-train_x[:,0], c='g', label='Idea Line')

    plt.scatter(train_x1[:,0], train_x1[:,1], marker='.', c='b')
    plt.scatter(train_x2[:,0], train_x2[:,1], marker='+', c='r')

    print(train_x.shape)

    plt.legend()

    plt.show()

    # scaler = StandardScaler()
    # scaler.fit(train_x)

    # train_x_std = scaler.transform(train_x)

    # model = ovo_svm(10, 10)


