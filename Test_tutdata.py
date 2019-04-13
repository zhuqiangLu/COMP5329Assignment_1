import numpy as np
import h5py
import matplotlib.pyplot as pl
from ipywidgets import interact, widgets
from matplotlib import animation
import comp5329
dataset = np.array([[-0.06465054,  0.37094156, -1.        ],
       [-0.80443548,  0.63068182, -1.        ],
       [-0.85604839, -0.33901515, -1.        ],
       [ 1.01922043, -0.10524892,  1.        ],
       [ 0.36545699, -0.58143939,  1.        ],
       [ 0.97620968, -0.6853355 ,  1.        ],
       [ 0.19341398, -0.96239177, -1.        ],
       [ 0.95040323, -1.45589827,  1.        ],
       [ 0.5030914 , -1.12689394,  1.        ],
       [-0.46034946, -1.12689394,  1.        ],
       [-0.15067204, -1.25676407,  1.        ],
       [-0.52056452, -0.04464286, -1.        ],
       [-0.3141129 ,  0.45752165, -1.        ],
       [-1.1141129 ,  0.26704545, -1.        ],
       [-1.1141129 ,  0.26704545, -1.        ],
       [-0.73561828,  0.43154762, -1.        ],
       [ 0.21061828,  1.23674242, -1.        ],
       [-0.46034946,  1.07224026, -1.        ],
       [ 0.18481183,  0.82115801, -1.        ],
       [-0.08185484, -0.183171  ,  1.        ],
       [ 0.30524194,  0.04193723, -1.        ],
       [-0.2108871 , -0.52949134,  1.        ],
       [-1.01948925, -0.89312771, -1.        ],
       [-0.70120968, -0.70265152,  1.        ],
       [-1.0625    , -1.41260823, -1.        ],
       [-0.89905914, -1.27408009,  1.        ],
       [ 1.23427419,  0.51812771, -1.        ],
       [ 0.82997312,  0.54410173,  1.        ],
       [ 0.78696237,  0.9163961 ,  1.        ],
       [ 0.56330645,  1.10687229,  1.        ]])



input_data = dataset[:,0:2].T
output_data = (dataset[:,2].astype(int) > 0)
train_data = np.eye(2)[output_data*1]
train_data = train_data.T
print(input_data.shape)
print(train_data.shape)

pl.figure(figsize=(6,6))
pl.scatter(input_data[:,0], input_data[:,1],c=[(['b', 'r'])[d>0] for d in output_data])
pl.title("Input Dataset")
pl.xlim((-2,2))
pl.ylim((-2,2))
#pl.show()

print("start training")
nn = comp5329.MLP([2,3,2], ['relu','relu','softmax'])
MSE = nn.fit(input_data, train_data, learning_rate=0.01, epochs=100)
predict_data = nn.predict(input_data)
print(predict_data)
print(train_data)
print('loss:%f'%MSE[-1])
