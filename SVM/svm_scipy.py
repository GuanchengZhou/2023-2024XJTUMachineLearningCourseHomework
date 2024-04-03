from sklearn.svm import SVC
from get_mnist import *
from sklearn.preprocessing import StandardScaler
from get_rand_data import *
import matplotlib.pyplot as plt

# class ovo_svm():
#     def __init__(self, cls_num, in_dim, C=1e9):
#         self.cls_num = cls_num
#         self.in_dim = in_dim
#         self.model = [ [LinearSVC(C) for j in range(cls_num) if i>j] for i in range(cls_num) ]
#         print(self.model)
#     def train(self, X, Y):
#         for i in range(self.cls_num):
#             for j in range(self.cls_num):
#                 if i>j:
#                     _X = X[(Y==i or Y==j), :]
#                     _Y = Y[(Y==i or Y==j)]
#                     self.model[i, j].fit(_X, _Y)
#     def predict(self, X, Y):
#         pass

        

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

    plt.plot(train_x, -p1/p2*train_x-b/p2, c='b')

    plt.scatter(train_x1[:,0], train_x1[:,1], marker='.', c='b')
    plt.scatter(train_x2[:,0], train_x2[:,1], marker='+', c='r')

    plt.show()

    # scaler = StandardScaler()
    # scaler.fit(train_x)

    # train_x_std = scaler.transform(train_x)

    # model = ovo_svm(10, 10)


