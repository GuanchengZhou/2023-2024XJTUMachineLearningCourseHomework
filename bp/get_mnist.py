from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
import torch as th

class mnist_loader:
    def __init__(self, batch_size, train_rate):
        mnist = loadmat("./dataset/mnist-original.mat/mnist-original.mat")
        self.mnist_data = mnist["data"].T
        self.mnist_data = (self.mnist_data-np.expand_dims(np.mean(self.mnist_data, axis=1), axis=1))/np.expand_dims(np.std(self.mnist_data, axis=1), axis=1)
        self.mnist_label = mnist["label"][0]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.mnist_data, self.mnist_label, test_size=1-train_rate)
        self.batchsize = batch_size
    def get_test_data(self):
        X = self.test_x
        Y = np.eye(10)[self.test_y.astype(int)]
        n = self.test_y.shape[0]
        complete_num = int(n/self.batchsize)
        batchs = []
        for i in range(complete_num):
            mini_batch_X = X[i*self.batchsize:(i+1)*self.batchsize, :]
            mini_batch_Y = Y[i*self.batchsize:(i+1)*self.batchsize, :]
            batchs.append((mini_batch_X.transpose(), mini_batch_Y.transpose()))
        return batchs
    def get_train_data(self):
        X = self.train_x
        Y = np.eye(10)[self.train_y.astype(int)]
        n = self.test_y.shape[0]
        complete_num = int(n / self.batchsize)
        batchs = []
        for i in range(complete_num):
            mini_batch_X = X[i * self.batchsize:(i + 1) * self.batchsize, :]
            mini_batch_Y = Y[i * self.batchsize:(i + 1) * self.batchsize, :]
            batchs.append((mini_batch_X.transpose(), mini_batch_Y.transpose()))
        return batchs


# print(type(mnist_data), mnist_data.shape)
# print(mnist_label.shape)

if __name__=='__main__':
    dataset = mnist_loader(128, .8)
    print(len(dataset.train_x), len(dataset.train_y), len(dataset.test_x), len(dataset.test_y))
    X, Y = dataset.get_train_data()
    print(X.shape, Y.shape)
    print(np.max(X))
    print(Y)
