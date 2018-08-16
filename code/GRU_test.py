import numpy as np
from keras.models import model_from_json
from keras.models import load_model
x_train = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test= np.load('../MAT/test_y.npy')
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
model=load_model("../Models/GRU.h5")
pred=model.evaluate(x=x_test[0:100],y=y_test[0:100],batch_size=25,verbose=1)
print("accuracy",pred[1])
print("loss",pred[0])
