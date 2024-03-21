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
    model = net(2, 1, [10, 10], alpha=alpha, bias=True)
    model.show_model()
    model.train()
    test_x, test_y = generate_xor_data(10, 0, 2)
    for iter in range(iter_num):
        lr = lr_0 * (decay_rate**(iter/decay_step))
        train_x, train_y = generate_xor_data(4, 0, 2)
        # train_x = np.array([[0,0],[1,0],[0,1],[1,1]]).transpose()
        # train_y = np.array([[0],[1],[1],[0]]).transpose()
        Loss = model.train_loop(train_x, train_y, lr)
        # if iter%10000==0:
        if iter % decay_step==0:
            print('Loss{} : {} {}'.format(iter, Loss, lr))

    model.test()
    test_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).transpose()
    _y = model.forward(test_x)
    print(_y.transpose())


if __name__=='__main__':
    xor_train(iter_num=100000, lr_0=.1, decay_rate=.98, decay_step=10000, alpha=0)