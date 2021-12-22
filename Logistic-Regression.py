import numpy as np
import matplotlib.pyplot as plt


def load_data(filename_entradas_x, filename_saidas_y):
  X = np.loadtxt(filename_entradas_x, delimiter=',', skiprows=0)
  Y = np.loadtxt(filename_saidas_y, delimiter=',', skiprows=0)  
  Y = np.expand_dims(Y, axis=1)
  return X, Y

def g(z):
  return 1/(1+np.exp(-z))

def calculate_h_theta(X, thetas, m):
  h_theta = np.zeros((m,1))
  h_theta = g(np.dot(X,thetas))
  return h_theta
  
def calculate_J(X, Y, thetas):
  m,_ = np.shape(X)
  J = 0
  h_theta = calculate_h_theta(X, thetas, m)
  J = (-1/m)*np.sum(np.multiply(Y,np.log(h_theta))+np.multiply((1-Y),np.log(1-h_theta)))
  b = np.sum(np.multiply(Y,np.log(h_theta)))
  return J, h_theta

def do_train(X, Y, thetas, alpha, iterations):
  J = np.zeros(iterations)
  m, n = np.shape(X)
  for i in range(iterations):
    J[i], h_theta = calculate_J(X, Y, thetas)
    h_theta = (calculate_h_theta(X,thetas,m))
    E = h_theta-Y
    thetas=thetas-np.multiply((alpha*(1/m)),(np.dot(np.transpose(X),E)))
  return J, thetas

def feature_scaling(X):
  m,n = np.shape(X) 
  mean_x = np.mean(X, axis=0)
  standard_deviation = np.std(X, axis=0)
  normalized_X = np.divide(X - mean_x, standard_deviation)
  return normalized_X

if __name__ == "__main__":
  X, Y = load_data('entradas_x.txt', 'saidas_y.txt')
  X = feature_scaling(X)
  X = np.insert(X, 0, values=1, axis=1)
  m, n = X.shape
  thetas = np.zeros((n,1))
  alpha = 0.01
  J,thetas = do_train(X, Y, thetas, alpha=alpha, iterations=10000)

  plt.plot(J)
  plt.title(r'$J(\theta$) vs iterações')
  plt.ylabel(r'$J(\theta$)', rotation=0)
  plt.xlabel("iteração")
  plt.show()
