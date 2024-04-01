import numpy as np
from gurobipy import *
import gurobipy as gp


class binary_svm:
    def __init__(self, in_dim, name=''):
        self.in_dim = in_dim
        self.model =Model("binary_svm_{}".format(name))
        self.w = []
        for i in range(in_dim):
            self.w.append(self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='w_{}'.format(i)))
        self.b = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')

    def train(self, x, y, n):
        """
        Train the binary SVM
        :param x: n * in_dim
        :param y: n * 1
        :return: None
        """
        N, _ = x.shape
        ep = []
        for i in range(N):
            ep.append(self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='epsilon_{}'.format(i)))

        self.model.addConstr()