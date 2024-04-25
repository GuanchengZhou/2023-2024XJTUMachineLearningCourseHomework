import pandas as pd
import numpy as np
import itertools
from kmeans_class import evaluate_external

names = {
        'setosa' : 0,
        'versicolor' : 1,
        'virginica' : 2,
    }
train_id = [30,128,1,30,9,108,140,132,78,56,79,60,129,68,28,5,109,18,44,124,10,84,103,126,88,123,132,60,128,72,20,141,127,54,91,12,45,98,61,77,92,33,114,113,141,21,86,33,109,22,12,128,26,80,114,40,67,90,138,90,89,86,145,86,63,108,93,118,31,113,70,21,126,98,58,13,33,3,86,57,19,126,136,98,53,41,0,135,71,136,91,70,79,64,128,12,81,17,53,127,73,135,8,147,117,128,62,114,80,54,143,122,48,71,117,10,35,137,92,26,]

def get_iris_data(train_rate=0.8):
    df = pd.read_csv('./iris.csv')
    X = np.array(df)[:, 1:-1]
    Y = np.array(df)[:, -1]
    Y = np.array([names[y] for y in Y])

    train_id = np.random.choice(len(X), size=int(len(X)*train_rate))
    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(len(X)):
        if i in train_id:
            train_x.append(X[i])
            train_y.append(Y[i])
        else:
            test_x.append(X[i])
            test_y.append(Y[i])

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


if __name__=='__main__':
    # Y2 = np.array([0,0,1,1,2,2])
    # Y1 = np.array([1,1,2,2,2,0])
    # Y1_ = np.array([2, 2, 0, 0, 0, 1])
    # evaluate_external(Y1, Y2,3)
    # evaluate_external(Y1_, Y2, 3)
    df = pd.read_csv('./iris.csv')
    #
    # print(df)
    #
    X = np.array(df)[:,1:-1]
    Y = np.array(df)[:,-1]
    Y = np.array([names[y] for y in Y])
    print(X[0])
    print(Y[:100])
    print(len(X),len(Y))
    # train_id = np.random.choice(len(X), size=15*8)
    # for i in train_id:
    #     print(i,end=',')