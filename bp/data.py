import numpy as np

def generate_xor_data(
    batch_size=100,
    low=0,
    high=100,
):
    x = []
    y = []
    for i in range(batch_size):
        a, b = np.random.randint(low, high), np.random.randint(low, high)
        x.append([a, b])
        y.append([a^b])
    x = np.array(x).transpose()
    y = np.array(y).transpose()
    return x, y

if __name__=='__main__':
    x, y = generate_xor_data(5, 0, 2)
    print(x.shape)
    print(y.shape)
