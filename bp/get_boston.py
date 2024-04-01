import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feature_names = [ 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age','dis',
                 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv' ]
feature_num = len(feature_names)

class boston_loader:
    def __init__(self, batch_size, train_rate):
        self.batchsize = batch_size
        df = pd.read_csv('./dataset/archive/BostonHousing.csv')
        df = df.dropna()
        data = df.to_numpy()
        self.data = np.random.shuffle(data)
        offset = int(data.shape[0] * train_rate)
        self.training_data = data[:offset]
        self.testing_data = data[offset:]
        print('train data', self.training_data.shape, 'test data', self.testing_data.shape)
        # 计算train数据集的最大值，最小值
        maximums, minimums = self.training_data.max(axis=0), self.training_data.min(axis=0)
        # 对数据进行归一化处理
        for i in range(feature_num):
            data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    def get_test_data(self):
        n = self.testing_data.shape[0]
        complete_num = int(n / self.batchsize)
        batchs = []
        for i in range(complete_num):
            mini_batch_X = self.testing_data[i * self.batchsize:(i + 1) * self.batchsize, :-1]
            mini_batch_Y = self.testing_data[i * self.batchsize:(i + 1) * self.batchsize, -1:]
            batchs.append((mini_batch_X.transpose(), mini_batch_Y.transpose()))
        return batchs
    def get_train_data(self):
        n = self.training_data.shape[0]
        complete_num = int(n / self.batchsize)
        np.random.shuffle(self.training_data)
        batchs = []
        for i in range(complete_num):
            mini_batch_X = self.training_data[i * self.batchsize:(i + 1) * self.batchsize, :-1]
            mini_batch_Y = self.training_data[i * self.batchsize:(i + 1) * self.batchsize, -1:]
            batchs.append((mini_batch_X.transpose(), mini_batch_Y.transpose()))
        return batchs


# print(type(mnist_data), mnist_data.shape)
# print(mnist_label.shape)

if __name__=='__main__':
    dataset = boston_loader(10, .8)
    train_batch = dataset.get_train_data()
    x, y = train_batch[0]
    print(x.shape, y.shape)
    test_batch = dataset.get_test_data()
    x, y = test_batch[0]
    print(x.shape, y.shape)
