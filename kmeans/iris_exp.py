from kmeans_class import *
from read_iris import *

if __name__=='__main__':
    # X, Y_gt = get_iris_data()
    train_x, train_y, test_x, test_y = get_iris_data()

    model = kmeans(k=3, distance_type='l2')

    model.train(train_x,lim_iter=1000)

    Y_pred = model.predict_kmeans(test_x)

    print('L2')
    evaluate_external(Y_pred, test_y, 3)

    model2 = kmeans(k=3, distance_type='l1')

    model2.train(train_x, lim_iter=1000)

    Y_pred2 = model2.predict_kmeans(test_x)

    print('L1')
    evaluate_external(Y_pred2, test_y, 3)