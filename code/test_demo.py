import librosa
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import json
x_train = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test = np.load('../MAT/test_y.npy')
root = "../test sample/"
file = "file1"

data, rate = librosa.load(root + file + ".wav")
mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=13)

temp3 = pad_sequences(mfcc, 4589, padding='pre',value=1)
temp4 = pad_sequences(mfcc, 4589, padding='post',value=1)
temp = np.array(temp3)
temp1=np.array(temp4)
nmodel = load_model("../Models/LSTM.h5")

# re-define model

yhat = nmodel.predict_classes(temp.reshape(1,temp.shape[0],temp.shape[1]), batch_size=1)
yhat1 = nmodel.predict_classes(temp.reshape(1,temp1.shape[0],temp1.shape[1]), batch_size=1)

with open('../map/data.txt') as json_file:
    data = json.load(json_file)
ans=0
for i in data.keys():
    if(str(yhat[0])==i):
        ans=data[i]
        break
print(ans)
for i in data.keys():
    if(str(yhat1[0])==i):
        ans=data[i]
        break
print(ans)
