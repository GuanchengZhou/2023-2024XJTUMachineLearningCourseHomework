import numpy as np

class net:
    def __init__(self, in_ch, out_ch, hidden_ch, alpha=1.):
        '''
            in_ch -> 128 -> hidden_ch -> 128 -> out_ch
            --linear--sig  --linear--sig--  --linear--sig-- --linear--
        '''
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_ch = hidden_ch
        self.alpha = alpha
        self.W = [np.random.randn(10, in_ch), np.random.randn(hidden_ch, 10), np.random.randn(10, hidden_ch), np.random.randn(out_ch, 10)]
        self.W = [np.array(w) for w in self.W]
        self.b = [np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()]
        
    def forward(self, x):
        print(type(self.W[0]))
        print(self.W[0])
        print(x.shape, self.W[0].shape)
        y1 = np.matmul(self.W[0], x)
        y2 = 1./(1+np.exp(-y1))
        
        y3 = np.matmul(self.W[1], y2)
        y4 = 1./(1+np.exp(-y3))
        
        y5 = np.matmul(self.W[2], y4)
        y6 = 1./(1+np.exp(-y5))
        
        y7 = np.matmul(self.W[3], y6)
        # y8 = 1./(1+np.exp(-y7))
        
        return y7

    def train(self, x, y):
        '''
        loss = abs(y7-y)^2 + lambda*(W1^2+W2^2+W3^2)
        '''
        y1 = np.matmul(self.W[0], x)
        y2 = 1./(1+np.exp(-y1))
        
        y3 = np.matmul(self.W[1], y2)
        y4 = 1./(1+np.exp(-y3))
        
        y5 = np.matmul(self.W[2], y4)
        y6 = 1./(1+np.exp(-y5))
        
        y7 = np.matmul(self.W[3], y6)
        # y8 = 1./(1+np.exp(-y7))
        
        Loss = (y7-y)**2 # + self.alpha*(np.sum(self.W**2))
        
        dLdy7 = 2*(y7-y)
        print(y.shape)
        print('y7 shape', dLdy7.shape, self.W[3].shape)
        dLdy6 = np.matmul(dLdy7, self.W[3])
        print('y6 shape', dLdy6.shape, y6.shape)
        dLdy5 = dLdy6 * (y6*(1-y6)).transpose()
        # print('y5 shape', dLdy5.shape, self.W[2].shape)
        dLdy4 = np.matmul(dLdy5, self.W[2])
        dLdy3 = dLdy4 * (y4*(1-y4)).transpose()
        dLdy2 = np.matmul(dLdy3, self.W[1])
        # dLdy1 = np.matmul(dLdy2, y2*(1-y2))
        dLdy1 = dLdy2 * (y2*(1-y2)).transpose()
        # dLdx  = np.matmul(dLdy1, self.W[0])
        
        dLdW = [np.einsum('ki,jk->ij',dLdy1, x), np.einsum('ki,jk->ij',dLdy3, y2), np.einsum('ki,jk->ij',dLdy5, y4)]
        
        print(dLdW[0].shape, dLdW[1].shape, dLdW[2].shape,)
        
        return y7

model = net(2, 1, 20)
x = np.array([[12,5],[1,3]])
y = np.array([[1,2]])
x = x.transpose((1,0))
print(x)
y = model.train(x, y)
print(y)
        
        