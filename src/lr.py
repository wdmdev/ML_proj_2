import numpy as np


def lr_l2(X: np.ndarray, y: np.ndarray, l: float):

  """
  Trains a linear regression model with l2 regularization penalty and returns weights
  X: Training set, n_samples x n_features
  y: Target, n_samples
  """

  M = X.shape[1]
  w = np.empty(M)
  y = y.squeeze()

  
  Xty = X.T @ y
  XtX = X.T @ X

  # Finding optimal weights using eq. 14.4 in the book
  w = np.linalg.solve(XtX + l * np.eye(M), Xty)

  return w
