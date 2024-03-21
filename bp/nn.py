import numpy as np


class Linear:
    def __init__(self, in_ch, out_ch, bias=False):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.W = np.random.random((out_ch, in_ch))
        self.dLdW = None
        self.bias = bias
        if self.bias:
            self.b = np.zeros((out_ch, 1))
            self.dLdb = None
        self.y = None
        self.x = None
        self.dydx = None
        self.train_flag = False

    def train(self):
        self.train_flag = True

    def test(self):
        self.train_flag = False

    def forward(self, x):
        """
        Forward Pass
        :param x: in_ch * Batch_size
        :return: out_ch * Batch_size
        """
        y = np.einsum('ij,jn->in', self.W, x)
        if self.bias:
            y = y + self.b
        if self.train_flag:
            self.y = y
            self.x = x
        return y

    def calc_grad(self, dLdy):
        """
        Calculate the gradient
        :param dLdy: 1 * out_ch * Batch_size
        :return: dLdx: 1 * in_ch * Batch_size
        """
        dLdx = np.einsum('kin,ij->kjn', dLdy, self.W)
        self.dLdW = np.einsum('kin,jn->ijn', dLdy, self.x)
        # print('dLdW')
        # print(np.sum(self.dLdW, axis=-1))

        if self.bias:
            self.dLdb = dLdy.transpose((1, 0, 2))
            # print('dLdb')
            # print(np.sum(self.dLdb, axis=-1))
        return dLdx

    def back_ward(self, lr, alpha=.001):
        self.W = self.W - lr * np.sum(self.dLdW, axis=-1) - 2*lr*alpha*self.W
        if self.bias:
            self.b = self.b - lr * np.sum(self.dLdb, axis=-1)

    def show(self):
        print('Linear\n in:{} out:{} bias:{}\n'.format(self.in_ch, self.out_ch, self.bias))


class Sigmoid:
    def __init__(self):
        self.train_flag = False
        self.x = None
        self.y = None

    def train(self):
        self.train_flag = True

    def test(self):
        self.train_flag = False

    def forward(self, x):
        # print('run')
        y = 1/(1+np.exp(-x))
        if self.train_flag:
            self.x = x
            self.y = y
        return 1/(1+np.exp(-x))

    def calc_grad(self, dLdy):
        """
        Calculate the gradient
        :param dLdy: 1 * c * Batch_size
        :return: dLdx: 1 * c * Batch_size
        """
        dLdx = np.einsum('kin,in->kin', dLdy, self.y*(1-self.y))
        return dLdx

    def back_ward(self, lr, alpha=1.):
        pass

    def show(self):
        print('Sigmoid\n')

class ReLU:
    def __init__(self):
        self.train_flag = False
        self.x = None
        self.y = None

    def train(self):
        self.train_flag = True

    def test(self):
        self.train_flag = False

    def forward(self, x):
        # print('run')
        y = (np.abs(x)+x)/2
        if self.train_flag:
            self.x = x
            self.y = y
        return 1/(1+np.exp(-x))

    def calc_grad(self, dLdy):
        """
        Calculate the gradient
        :param dLdy: 1 * c * Batch_size
        :return: dLdx: 1 * c * Batch_size
        """
        # dLdx = np.einsum('kin,in->kin', dLdy, self.y*(1-self.y))
        dd = np.zeros_like(self.x)
        dd[self.x>0] += 1
        dLdx = np.einsum('kin,in->kin', dLdy, dd)
        return dLdx

    def back_ward(self, lr, alpha=1.):
        pass

    def show(self):
        print('ReLU\n')

if __name__=='__main__':
    print('Test Linear')
    block = Linear(2, 4, bias=True)
    x = np.random.randn(2, 3)
    block.train()
    y = block.forward(x)
    print(y.shape)
    dLdy = np.random.randn(1, 4, 3)
    block.calc_grad(dLdy)

    print('Test Sigmoid')
    block = Sigmoid()
    block.train()
    y = block.forward(x)
    print(y.shape)
    dLdy = np.random.randn(1, 2, 3)
    block.calc_grad(dLdy)
