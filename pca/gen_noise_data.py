import numpy as np
import matplotlib.pyplot as plt

def origin_data(alphas, deltas, powers, time_steps=1000, xlim=1):
    assert len(alphas)==len(deltas) and len(alphas)==len(powers), 'The numbers of alpha, delta and powers are not similar'
    X = np.linspace(0, xlim, time_steps)
    Y = np.zeros_like(X)
    for i in range(len(alphas)):
        alpha, delta, power = alphas[i], deltas[i], powers[i]
        Y += power * np.cos(alpha*X+delta)
    return X, Y

def add_noise(Y, power):
    noise = np.random.randn(len(Y))
    return Y + power*noise

if __name__=='__main__':
    alphas = np.array([20,30,50])
    powers = np.array([1, 2, 3])
    deltas = np.array([0, 0, 0])
    X, Y = origin_data(alphas, deltas, powers)

    # Y = add_noise(Y, 0.8)
    plt.plot(X, Y)
    plt.show()
