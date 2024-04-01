from nn import *

class net:
    def __init__(self, in_ch, out_ch, layers, bias=False, alpha=.001, reg=False):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layers = []
        self.alpha = alpha
        if len(layers)==1:
            c = layers[0]
            self.layers.append(
                Linear(in_ch, c, bias)
            )
            self.layers.append(
                Sigmoid()
            )
            self.layers.append(
                Linear(c, out_ch, bias)
            )
            # self.layers.append(
            #     Sigmoid()
            # )
        else:
            self.layers.append(
                Linear(in_ch, layers[0], bias)
            )
            self.layers.append(
                Sigmoid()
            )
            for i in range(len(layers)-1):
                self.layers.append(
                    Linear(layers[i], layers[i+1], bias)
                )
                self.layers.append(
                    Sigmoid()
                )
            self.layers.append(
                Linear(layers[-1], out_ch, bias)
            )

    def train(self):
        for layer in self.layers:
            layer.train()

    def test(self):
        for layer in self.layers:
            layer.test()

    def forward(self, x):
        """
        forward pass
        :param x: in_ch * N
        :return: out_ch * N
        """
        y = x
        for layer in self.layers:
            # print(y.shape)
            print(y)
            y = layer.forward(y)
        return y

    def train_loop(self, x, y, lr):
        """
        train model through x
        :param x: in_ch * N
        :param y: 1 * N
        :return: Loss
        """
        _y = x
        for layer in self.layers:
            _y = layer.forward(_y)
        L = np.sum(.5*(_y-y)**2)
        dLdy = np.expand_dims((_y-y), axis=0)
        for i in range(len(self.layers)):
            layer = self.layers[len(self.layers)-i-1]
            dLdy = layer.calc_grad(dLdy)
            layer.back_ward(lr, self.alpha)
        return np.mean(L)

    def show_model(self):
        for layer in self.layers:
            layer.show()


if __name__ == '__main__':
    # model = net(2, 1, [10, 20, 10], )
    # model.show_model()
    # x = np.random.randn(2, 10)
    # model.train()
    # y = model.forward(x)
    # yy = np.random.randn(1, 10)
    # L = model.train_loop(x, yy, lr=1e-3)
    # print(L)
    model = net(2, 1, [2], alpha=0., bias=True)
    model.show_model()
    model.train()
    model.layers[0].W = np.array([
        [3.77876719, 5.77131988],
        [3.78345326, 5.79331783],
    ]).transpose()
    model.layers[0].b = np.array([[-5.79319383, -2.42207539]]).transpose()
    model.layers[2].W = np.array([
        [-8.17262084],
        [7.55637468]]).transpose()
    model.layers[2].b = np.array([[-3.41823076]])
    model.train()
    test_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).transpose()
    test_y = np.array([[0], [1], [1], [0]]).transpose()
    L = model.train_loop(test_x, test_y, 1e-1)
    print(L.transpose())

    # print('Linear0', model.layers[0].y.shape)
    # print(model.layers[0].y[:,0])
    for epoch in range(10):
        loss = model.train_loop(test_x, test_y, .1)
        print(epoch, loss)
