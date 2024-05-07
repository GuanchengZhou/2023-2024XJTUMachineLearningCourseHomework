import os

import cv2

from mnist import *
from pca import *

powers = [0.25, 0.5, 0.75, 1.0]
n_comps = [0.25, 0.5, 0.8, 0.9]
# n_comps = [1, 10, 50]

def run(power, n_comp):
    dataset = mnist_loader(0.8)
    X = dataset.train_x
    noise_X = dataset.add_noise(X, power)
    print(noise_X.shape)
    # dataset.show_img(noise_X[0])
    pca = PCA(n_comp)
    pca.fit(noise_X)

    test = dataset.add_noise(dataset.test_x, power)
    # test = noise_X

    dataset.gen_result(test, 'origin_{}_{}'.format(int(power*100), int(n_comp) if n_comp != 'mle' else 'mle'), './result4')
    DD = pca.transform(test)
    Y = pca.inverse_transform(DD)
    print(Y.shape)
    dataset.gen_result(Y, 'predict_{}_{}'.format(int(power*100), int(n_comp) if n_comp != 'mle' else 'mle'), './result4')

    # features = pca.components_
    # print(features.shape)
    # for i, feature in enumerate(features):
    #     image = feature.copy()
    #     image = dataset.unstd_img(image)
    #     image = 255 * image
    #     image = image.astype(np.int32)
    #     image = (image - np.min(image)) / np.max(image) * 255
    #     image = image.astype(np.int32)
    #     cv2.imwrite(os.path.join('./result3/feature10', 'feature_{}_{}_{}.png'.format(int(power*100), int(n_comp) if n_comp != 'mle' else 'mle', i)), image)

if __name__=='__main__':
    run(powers[1], 10)
    for power in powers:
        for n_comp in n_comps:
            run(power, n_comp)
        # run(power, 'mle')
