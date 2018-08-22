import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing.sequence import TimeseriesGenerator
import math
from keras.preprocessing.sequence import pad_sequences

# Load dataset
df=pd.read_csv('../speakers_all.csv')
path="../recordings_new/"
filename=df['filename']
native=df['native_language']
native=list(set(native))
filename=np.array(filename)
# Get a dict of label and native
x_train,y_train,x_test,y_test=[],[],[],[]
token=[]
nat=df['native_language']
nat=np.array(nat)
for i in range(0,len(native)):
    token.append(i)
native=np.array(native)
for i in tqdm(range(len(df))):
    try:
        fi=(path+filename[i]+".wav")
        data,rate=librosa.load(fi)
        data=data[len(data)-30000:len(data)]
        mfcc = librosa.feature.mfcc(y=data,sr=rate, n_mfcc=20)
        for j in range(0,len(native)):
            if(nat[i]==native[j]):
                y_train.append(token[j])
                x_train.append(mfcc)
               
    except:
        print("error")
x_train=np.array(x_train)
print(x_train.shape)
y=np_utils.to_categorical(y_train, 214)
y_train,y_test,y_val=y[0:1800],y[1800:1937],y[1937:2137]
x_train,x_test,x_val=x_train[0:1800],x_train[1800:1937],x_train[1937:2137]
print(x_train.shape,x_test.shape,x_val.shape)
print(y_train.shape,y_test.shape,y_val.shape)
np.save('../MAT/train_x.npy', x_train)
np.save('../MAT/train_y.npy', y_train)
np.save('../MAT/test_x.npy', x_test)
np.save('../MAT/test_y.npy', y_test)
np.save('../MAT/val_x.npy', x_val)
np.save('../MAT/val_y.npy', y_val)
