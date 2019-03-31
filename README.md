# comp5329Assignment1 Edit Record

The code in 3-MLP is used as the template of this assignment 
There are some changes to the original 3-MLP

## 31-Mar:
### Added: 
  #### L1 regularization
    loss <- loss + lamda/m(|W|)
  #### L2 regularization
    loss <- loss + lamda/m(|W|^2)
  #### dropout regularization
    forward: calculate the mask layer for a, apply mask to a, scale a by the keep-prob
    backward: use the mask applied in forward, apply the same mask to the corresponding a before calculate da/dz
  #### Cross Entropy 
    ce = 1/m(sum(y log (y_hat )))
  #### OneHot method
    to covert a multi-class matrix to a one hot matrix
    example:
      y = [0, 1, 2]
      y_one_hot = [[1,0,0],
                   [0,1,0],
                   [0,0,1]]
### Changes:
  
  #### The expected dimension of input data and output data
    before:
      X is (n_example, n_features)
      Y is (n_example, n_classes)
      W is (n_in, n_out)
      b is (n_out,)
    after:
      X is (n_features, n_exampes)
      Y is ( n_classes, n_examples)
      W is (n_out, n_in)
      b is (n_out, 1)
  #### Way to calculate Z for each layer
    before: 
      Z = X.T * W + b
    after:
      Z = W * X + b

