import math

from data import *
from net import *

def xor_train(
    iter_num=1000,
    lr_0=1e-3,
    decay_rate=.8,
    decay_step=100,
    alpha=1.,
):
    # np.random.seed(42)
    # W1 = np.random.random((2, 2))
    # b1 = np.zeros((1, 2))
    # W2 = np.random.random((2, 1))
    # b2 = np.zeros((1, 1))
    #
    # print('W1-----')
    # print(W1)
    # print('b1-----')
    # print(b1)
    #
    # print('W2-----')
    # print(W2)
    # print('b2-----')
    # print(b2)

    model = net(2, 1, [2], alpha=alpha, bias=True)
    model.show_model()
    model.train()
    # model.layers[0].W = W1.transpose()
    # model.layers[0].b = b1.transpose()
    # model.layers[2].W = W2.transpose()
    # model.layers[2].b = b2.transpose()

    test_x, test_y = generate_xor_data(10, 0, 100)
    cnt = 0
    for iter in range(10000):
        lr = lr_0 * (decay_rate**(iter/decay_step))
        # train_x, train_y = generate_xor_data(4, 0, 2)
        # print(train_x)
        # train_x = np.array([[0,0],[1,0],[0,1],[1,1]]).transpose()
        # train_y = np.array([[0],[1],[1],[0]]).transpose()
        test_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).transpose()
        test_y = np.array([[0], [1], [1], [0]]).transpose()
        Loss = model.train_loop(test_x, test_y, .1)
        # if iter%10000==0:
        cnt += 1
        if iter % 1000 ==0:
            print('Loss{} : {} {}'.format(iter, Loss, lr))

    model.test()
    test_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).transpose()
    _y = model.forward(test_x)
    print(_y.transpose())


if __name__=='__main__':
    xor_train(iter_num=100000, lr_0=5e-4, decay_rate=0.9, decay_step=1000, alpha=0.)