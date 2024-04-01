import math

from get_mnist import *
from net_mnist import *

from matplotlib import pyplot as plt

def calc_pr(pred_y, gt_y):
    pred_y = np.argmax(pred_y, axis=0)
    gt_y = np.argmax(gt_y, axis=0)
    pr = np.sum(pred_y==gt_y)/len(pred_y)
    return pr

def mnist_train(
    iter_num=1000,
    lr_0=1e-3,
    batch_size=128,
    train_rate=.8,
    decay_rate=.8,
    decay_step=100,
    alpha=1.,
    log_step=100,
):
    dataset = mnist_loader(batch_size, train_rate)
    model = net(784, 10, [10, 128, 10], alpha=alpha, bias=True)
    model.show_model()
    model.train()
    save_step = int(iter_num/100)
    X = []
    losses = []
    test_batchs = dataset.get_test_data()
    # print('run')
    step = 0
    for iter in range(iter_num):
        train_batchs = dataset.get_train_data()
        print(len(train_batchs))
        for train_x, train_y in train_batchs:
            # print(step)
            lr = lr_0 * (decay_rate**(step/decay_step))
            # print(train_x.shape, train_y.shape)
            # train_x, train_y = dataset.get_train_data()
            # train_x = np.array([[0,0],[1,0],[0,1],[1,1]]).transpose()
            # train_y = np.array([[0],[1],[1],[0]]).transpose()
            Loss = model.train_loop(train_x, train_y, lr)
            if step % save_step == 0:
                losses.append(Loss)
                X.append(iter)
            step += 1
        print('train: {}'.format(Loss))
        Ls = []
        Prs = []
        for test_x, test_y in test_batchs:
            _y = model.forward(test_x)
            pr = calc_pr(test_y, _y)
            # print(_y.transpose())
            Loss = model.train_loop(test_x, test_y, 0.1)
            Ls.append(Loss)
            Prs.append(pr)
        print('Loss{}: {} {}'.format(iter, np.mean(Ls), np.mean(Prs)))
        # if iter%10000==0:
        # if iter % log_step==0:
        #     model.test()
        #
        #     _y = model.forward(train_x)
        #     pr = calc_pr(_y, train_y)
        #     print('Loss{} : {} {} {}'.format(iter, Loss, pr, lr))
            # Ls = []
            # Prs = []
            # for test_x, test_y in batchs:
            #     _y = model.forward(test_x)
            #     pr = calc_pr(test_y, _y)
            #     # print(_y.transpose())
            #     Loss = model.train_loop(test_x, test_y, 0.1)
            #     Ls.append(Loss)
            #     Prs.append(pr)
            # print(np.mean(Ls), np.mean(Prs))
        # if iter % save_step == 0:
        #     losses.append(Loss)
        #     X.append(iter)

    # model.tr()

    Ls = []
    Prs = []
    for test_x, test_y in test_batchs:
        _y = model.forward(test_x)
        pr = calc_pr(test_y, _y)
        # print(_y.transpose())
        Loss = model.train_loop(test_x, test_y, 0.1)
        Ls.append(Loss)
        Prs.append(pr)
    print(np.mean(Ls), np.mean(Prs))

    # plt.plot(X,losses)
    # plt.show()

if __name__=='__main__':
    mnist_train(iter_num=1000, lr_0=1, batch_size=128, decay_rate=.98, decay_step=1000, alpha=0, log_step=1000)