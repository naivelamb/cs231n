import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(0, X.shape[0]):
      score = np.matmul(X[i, :], W)
      score -= np.max(score)
      score_exp = np.exp(score)
      loss += -np.log(score_exp[y[i]]/np.sum(score_exp))
      dW += X[i, None].T.dot(score_exp[None,:])/np.sum(score_exp)
      dW[:, y[i]] -= X[i,:].T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  # divide by train number
  loss /= X.shape[0]
  dW /= X.shape[0]
  # add regularization term
  loss += reg * np.sum(W*W)
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = np.matmul(X, W)
  stab = np.amax(score, axis = 1)
  score -= stab.reshape(num_train, 1)
  softmax = np.exp(score) / np.sum(np.exp(score), axis=1)[:, None]
  
  loss = -np.sum(np.log(softmax[range(num_train), y]))
  loss /= num_train
  loss += reg*np.sum(W*W)
  
  margin = softmax
  margin[np.arange(num_train), y] += -1
  dW= X.T.dot(margin)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

