import matplotlib.pyplot as plt
import numpy as np

def get_random_data(mu=(0,0), sigma=(1,1), n=10, label=1):
    X = [(np.random.normal(mu[0], sigma[0]), np.random.normal(mu[1], sigma[1])) for i in range(n)]
    Y = [label for i in range(n)]
    return np.array(X), np.array(Y)

if __name__=='__main__':
    X1, Y1 = get_random_data((0,0), (1,1), n=25)
    plt.scatter(X1[:,0], X1[:,1], marker='.', c='b')
    X2, Y2 = get_random_data((2, 2), (1, 1), n=25)
    plt.scatter(X2[:,0], X2[:,1], marker='+', c='r')
    plt.plot(X1[:,0],2-X1[:,0], c='g', label='Idea Line')
    plt.plot(X2[:, 0], 2 - X2[:, 0], c='g')
    plt.legend()
    plt.show()