import os.path

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split

class mnist_loader:
    def __init__(self, train_rate):
        mnist = loadmat("../dataset/mnist-original.mat/mnist-original.mat")
        self.mnist_data = mnist["data"].T
        self.mnist_data = (self.mnist_data-np.expand_dims(np.mean(self.mnist_data, axis=1), axis=1))/np.expand_dims(np.std(self.mnist_data, axis=1), axis=1)
        self.mnist_label = mnist["label"][0]
        self.mnist_mu, self.mnist_std = 0.1307, 0.3081
        if train_rate!=1:
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.mnist_data, self.mnist_label, test_size=1-train_rate)
        else:
            self.train_x, self.train_y = self.mnist_data, self.mnist_label
            self.test_x, self.test_y = np.array([]), np.array([])
        # self.batchsize = batch_size
    def get_test_data(self):
        X = self.test_x
        # Y = np.eye(10)[self.test_y.astype(int)]
        Y = self.test_y
        return X, Y
    def get_train_data(self):
        X = self.train_x
        # Y = np.eye(10)[self.train_y.astype(int)]
        Y = self.train_y
        return X, Y

    def add_noise(self, X, power):
        # print(X.shape)
        noise = np.random.normal(size=X.shape)
        noise = noise * power
        return X + noise
    def unstd_img(self, img):
        img = np.resize(img, (28, 28))
        img = self.mnist_std * img + self.mnist_mu
        img[img <= 0] = 0
        img[img >= 1] = 1
        return img
    def show_img(self, img):
        img = self.unstd_img(img)
        plt.imshow(img)
        plt.show()
    def save_img(self, X, proc_func,  dir='./'):
        save_path = os.path.join(dir, 'result')
        try:
            os.mkdir(os.path.join(save_path, 'old'))
            os.mkdir(os.path.join(save_path, 'new'))
        except:
            pass
        for i, x in enumerate(X):
            x_o = x.copy()
            x_n = proc_func(x)
            cv2.imwrite(os.path.join(save_path, 'old', 'img_{}.png'.format(i)), x_o)
            cv2.imwrite(os.path.join(save_path, 'new', 'img_{}.png'.format(i)), x_n)

    def gen_result(self, X, prefix='', dir='./'):
        image_path = os.path.join(dir, prefix+'.png')
        new_image = Image.new("L", (28 * 3, 28 * 6), 0)
        images = [[None for i in range(3)] for j in range(6)]
        for i, image_vector in enumerate(X):
            if i==18:
                break
            image = image_vector.copy()
            image = self.unstd_img(image)
            image = 255 * image
            image = image.astype(np.int32)
            # print(image.shape, np.max(image), np.min(image))
            x = i // 6
            y = i % 6
            # print(x, y)
            image = Image.fromarray(image)
            new_image.paste(image, (x * 28, y * 28))
        # new_image = new_image.convert('RGB')
        new_image.save(image_path)

# print(type(mnist_data), mnist_data.shape)
# print(mnist_label.shape)

if __name__=='__main__':
    dataset = mnist_loader(1)
    print(len(dataset.train_x), len(dataset.train_y), len(dataset.test_x), len(dataset.test_y), )
    print(np.max(dataset.train_x), np.mean(dataset.train_x), np.std(dataset.train_x))
    # X = dataset.train_x
    # X = dataset.mnist_std * X + dataset.mnist_mu
    # X[X>=1] = 1
    # X[X<=0] = 0
    # print(np.min(X), np.max(X))

    # dataset.show()

