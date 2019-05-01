from Activation import tanh, sigmoid, softmax, relu
from BatchNormalizer import standard
from Initializer import Xavier, He
from Cost import Cross_Entropy
from Dropout import Dropout
from Optimizer import Momentum, Nesterov, AdaGrad, AdaDelta, RMSProp, Adam
from Regularizer import L1, L2
from Model import Model
import DataPreprocessor
import config
import h5py
import numpy as np


def get_batch_size():
    return config.Batch_Size

def get_drop():
    return config.Dropout_Rate

def get_regularizer():
    decay_rate = config.Weight_Decay

    if(config.Regularizer == "L2"):
        return L2(decay_rate)
    elif(config.Regularizer == "L1"):
        return L1(decay_rate)
    else:
        return None

def get_norm():
    if(config.Batch_Normalization):
        return standard()
    else:
        return None

def get_opt():
    opt = config.OPtimization.lower()
    if(opt == "adam"):
        return Adam()
    elif(opt == "adadelta"):
        return Adadelta()
    elif(opt == "adagrad"):
        return AdaGrad()
    elif(opt == "rmsprop"):
        return RMSProp()
    elif(opt == "nesterov"):
        return Nesterov()
    elif(opt == "momentum"):
        return Momentum()
    else:
        return None





def train_and_predict():

    X, Y, toPredict= DataPreprocessor.get_preprocessed_data(config.Data_Path)

    data = DataPreprocessor.train_cv_test_split(X, Y, config.Training_Rate, config.Cross_Validate_Rate, config.Test_Rate)
    (train_X, train_Y, cv_X, cv_Y, test_X, test_Y) = data
    model = Model(train_X, train_Y, batch_size = get_batch_size() , drop = get_drop(), regularizer = get_regularizer(), norm = get_norm(), optimizer = get_opt())
    model.print_Info(config.Print_Info, config.Print_At)
    model.cross_validate(cv_X, cv_Y)
    model.add_layer(192, ini = He(), acti = relu())
    model.add_layer(96, ini = He(), acti = relu())
    model.add_layer(48, ini = He(), acti = relu())
    model.add_last_layer(ini= He())
    model.fit(epoch = config.Epoch, learning_rate = config.Learning_Rate)
    model.test(test_X, test_Y)
    model.plot(config.Plot_Loss, config.Plot_Accuracy)

    predict = model.predict(toPredict).T
    predict = np.argmax(predict, axis = 1)
    f = h5py.File(config.Save_To + "/Predicted_labels.h5",'a')
    f.create_dataset('/predicted_label',data = predict, dtype = np.float32)
    f.close()


train_and_predict()
