import numpy as np
import h5py



def load_data(path):
    if(path is None):
        path = "."
    with h5py.File(path + '/train_128.h5','r') as H:
        data = np.copy(H['data'])
    with h5py.File(path +'/train_label.h5','r') as H:
        label = np.copy(H['label'])
    with h5py.File(path + '/test_128.h5','r') as H:
        test = np.copy(H['data'])

    return data,label,test

def standardize_data(X):
    mean = np.mean(X, axis = 1, keepdims = True)
    var = np.var(X, axis = 1, keepdims = True)
    return (X-mean)/np.sqrt(var + 1e-8)

def train_cv_test_split(X, Y, train_rate, cv_rate, test_rate):

    m = X.shape[1]
    train = round(m*train_rate)
    cv = train + round(m*cv_rate)
    test = cv + round(m*test_rate)

    train_X = X[:,:train]
    train_Y = Y[:,:train]
    if(train_rate == 0):
        train_X = None

    cv_X = X[:,train:cv]
    cv_Y = Y[:, train:cv]
    if(cv_rate == 0):
        cv_X = None

    test_X = X[:, cv:]
    test_Y = Y[:, cv:]
    if(test_rate == 0):
        test_X = None



    return (train_X, train_Y, cv_X, cv_Y, test_X, test_Y)

def one_hot(label):
    return np.eye(np.max(label)+1)[label]

def get_preprocessed_data(path = None):
    #data preprocessing
    #load data
    print("<==============Loading Data===============>")
    X, Y, test = load_data(path)
    print("Trian data shape: {}".format(X.shape))
    print("Trian label shape: {}".format(Y.shape))
    print("Test data shape: {}".format(test.shape))
    print("<=============Loading Preprocessing=======>")
    Y = one_hot(Y)
    print("one_hot label")
    X = X.T
    Y = Y.T
    test = test.T
    #standardize data
    X = standardize_data(X)
    test = standardize_data(test)
    print("standardized training data and toPredict data")
    print("<===========After Data Preprocessing======>")
    print("trian data shape: {}".format(X.shape))
    print("trian label shape: {}".format(Y.shape))
    print("test data shape: {}".format(test.shape))
    return X, Y, test
