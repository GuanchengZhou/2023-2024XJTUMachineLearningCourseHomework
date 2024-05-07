import numpy as np
from gen_noise_data import *
from sklearn.decomposition import PCA

class pca:
  def __init__(self, k):
    self.k = k
    self.features = None
    self.mean = None
  def fit(self, X):
    class Cmp(tuple):
      def __lt__(self, other):
        return self[0] < other[0]
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    self.mean = mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    print(eig_val.shape, eig_vec.shape)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    # eig_pairs.sort(reverse=True)
    eig_pairs.sort(key=Cmp)
    # select the top k eig_vec
    self.feature = np.array([ele[1] for ele in eig_pairs[:self.k]]) # k * d
    # print('feature', self.feature.shape)
    # get new data
    # data = np.dot(norm_X, np.transpose(self.feature))
    # print('data', data.shape)
  def transform(self, X):
    norm_X = X - self.mean
    data = np.dot(norm_X, np.transpose(self.feature))
    return data
  def inverse_transform(self, X):
    assert self.feature is not None, 'Please train first'
    data = np.einsum('nk,kd->nd', X, self.feature) + self.mean
    return data
# def pca(X,k):#k is the components you want
#   #mean of each feature
#   n_samples, n_features = X.shape
#   mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
#   #normalization
#   norm_X=X-mean
#   #scatter matrix
#   scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
#   #Calculate the eigenvectors and eigenvalues
#   eig_val, eig_vec = np.linalg.eig(scatter_matrix)
#   eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
#   # sort eig_vec based on eig_val from highest to lowest
#   eig_pairs.sort(reverse=True)
#   # select the top k eig_vec
#   feature=np.array([ele[1] for ele in eig_pairs[:k]])
#   print('feature', feature.shape)
#   #get new data
#   data=np.dot(norm_X, np.transpose(feature))
#   print('data', data.shape)
#   return data

if __name__=='__main__':
  alphas = np.array([20, 30, 50])
  powers = np.array([10, 20, 30])
  deltas = np.array([0, 0, 0])

  model = pca(2)

  X = np.array([[-1, 1, 2], [-2, -1, 4], [-3, -2, 0], [1, 1, 1], [2, 1, -1], [3, 2, 2]])
  model.fit(X)
  D = model.transform(X)
  print('transform', D.shape)
  X_inverse = model.inverse_transform(D)
  print('inverse', X_inverse.shape)

  # pca = PCA(n_components=1)
  # pca.fit(X)
  # print(pca.transform(X))

  # X, Y = origin_data(alphas, deltas, powers)
  # Y = add_noise(Y,0.5)
  # D = [[X[i], Y[i]] for i in range(len(X))]
  # D = np.array(D)
  # plt.cla()
  # plt.plot(D[:, 0], D[:, 1])
  # plt.show()
  #
  # pca = PCA(n_components=1)
  # print(D.shape)
  # pca.fit(D)
  # X_reduction = pca.transform(D)
  # X_restore = pca.inverse_transform(X_reduction)
  #
  # plt.cla()
  # plt.plot(X_restore[:,0], X_restore[:,1])
  # plt.show()
#---------------------------------------------
  n = 1000
  D = []
  for i in range(2000):
    alphas = np.array([20, 30, 50])
    powers = np.array([1, 2, 3])
    deltas = np.array([0, 0, 0])
    X, Y = origin_data(alphas, deltas, powers)
    Y = add_noise(Y, 0.8)
    D.append(Y)
  D = np.array(D)

  plt.cla()
  plt.plot(X, D[0, :])
  plt.show()

  # D = D.transpose()
  print(D)
  print(D.shape)

  pca = PCA(1)
  pca.fit(D)
  DD = pca.transform(D)
  DD = pca.inverse_transform(DD)

  X, Y = origin_data(alphas, deltas, powers)
  plt.cla()
  plt.plot(X, DD[0,:])
  # plt.plot(X, Y, c='r')
  plt.show()

  loss = np.mean((DD[0,:] - Y)**2)
  print(loss)
