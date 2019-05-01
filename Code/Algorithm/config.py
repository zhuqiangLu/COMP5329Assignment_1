'''
Please note that this config serves as a GUI

'''

#Path

#if the data folder is in the root, please be empty string
Data_Path = "../Assignment-1-Dataset"

#if the save to folder is in the root, please be empty string
Save_To = "../Assignment-1-Output"

#Hyperparameters
Learning_Rate = 0.0005          #please be greater than 0
Epoch = 50                      #please be greater than 0
Batch_Size = 32                 #please be greater than 0
Dropout_Rate = 0.3              #please be 0 to 1 (inclusive)
Weight_Decay = 0                #please be 0 to 1 (inclusive)


Regularizer = None              #please be L1 or L2 in string
Batch_Normalization = True      #please be boolean
OPtimization = "adam"           #please be optimization in string
#Available optimization include:
# Adam, AdaDelta, RMSProp, AdaGrad, Nesterov, Momentum


Training_Rate = 1              #please be 0 to 1 (inclusive)
Cross_Validate_Rate = 0        #please be 0 to 1 (inclusive)
Test_Rate = 0                   #please be 0 to 1 (inclusive)


Plot_Loss = True                #please be boolean
Plot_Accuracy = True            #please be boolean
Print_Info = True               #please be boolean
Print_At = 1                    #please be int and be greater than 0
