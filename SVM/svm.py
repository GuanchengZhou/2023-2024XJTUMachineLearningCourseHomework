import numpy as np
from gurobipy import *
import gurobipy as gp


class binary_svm:
    def __init__(self, in_dim, name=''):
        self.in_dim = in_dim
        self.model =Model("binary_svm_{}".format(name))
        self.w = []
        self.w_val = np.array([0 for i in range(in_dim)])
        for i in range(in_dim):
            self.w.append(self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='w_{}'.format(i)))
        self.b = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
        self.b_val = 0

    def train(self, x, y, c=1.0):
        """
        Train the binary SVM
        :param x: n * in_dim
        :param y: n
        :return: None
        """
        N, _ = x.shape
        ep = []
        for i in range(N):
            ep.append(self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='epsilon_{}'.format(i)))

        for i in range(N):
            val = self.b
            for j in range(self.in_dim):
                val += self.w[j] * x[i][j]
            self.model.addConstr(y[i]*val >= 1-ep[i])

        object_eq = 0
        for i in range(self.in_dim):
            object_eq += .5*self.w[i] * self.w[i]
        for i in range(N):
            object_eq += c*ep[i]
        self.model.setObjective(object_eq, GRB.MINIMIZE)

        self.model.optimize()

        for i in range(self.in_dim):
            self.w_val[i] = self.w[i].x
        self.b_val = self.b.x

        print(self.w_val)
        print(self.b_val)

    def predict(self, x):
        """
        Predict
        :param x: N * in_dim
        :return:
        """
        N, _ = x.shape
        bs = np.expand_dims(self.b, axis=0).repeat(N) # N
        result = np.einsum('i,ki->k', self.w, self.x) + bs
        result[result<0] = -1
        result[result>=0] = 1
        return result

    def test(self, x, y):
        N, _ = x.shape
        bs = np.expand_dims(self.b, axis=0).repeat(N)  # N
        result = np.einsum('i,ki->k', self.w, self.x) + bs
        result[result < 0] = -1
        result[result >= 0] = 1
        pr = np.sum(result==y)/N
        print('Precision:', pr)

if __name__ == '__main__':
    n, dim = 10, 3
    x = np.random.randn(n, dim)
    y = np.random.randn(n)
    for i in range(len(y)):
        if y[i] > 0:
            y[i] = 1
        else:
            y[i] = -1

    print(x.shape, y.shape)

    model = binary_svm(dim)
    model.train(x, y, 1)



