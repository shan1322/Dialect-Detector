
# coding: utf-8

# In[1]:


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


# In[2]:


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
print(nat)


# In[3]:


for i in range(0,len(native)):
    token.append(i)
native=np.array(native)


# In[4]:


for i in tqdm(range(len(df))):
    try:
        fi=(path+filename[i]+".wav")
        rate,data=librosa.load(fi)
        mfcc = librosa.feature.mfcc(y=rate,sr=data, n_mfcc=20)
        for j in range(0,len(native)):
            if(nat[i]==native[j]):
                y_train.append(token[j])
                x_train.append(mfcc)

    except:
        1


# In[6]:


temp=[]
for j in x_train:
    temp.append(j.shape[1])
print(int((min(temp)+max(temp))/2))


# In[7]:


X_train=[]
for i in x_train:
    X_train.append(pad_sequences(i, maxlen=2634))


# In[8]:


X_train=np.array(X_train)


# In[9]:


X_train.shape


# In[10]:


y=np_utils.to_categorical(y_train, 214)
y_train,y_test,y_val=y[0:1800],y[1800:1937],y[1937:2137]
x_train,x_test,x_val=X_train[0:1800],X_train[1800:1937],X_train[1937:2137]
print(x_train.shape,x_test.shape,x_val.shape)
print(y_train.shape,y_test.shape,y_val.shape)


# In[12]:


np.save('../MAT/train_x.npy', x_train) 
np.save('../MAT/train_y.npy', y_train) 
np.save('../MAT/test_x.npy', x_test) 
np.save('../MAT/test_y.npy', y_test) 
np.save('../MAT/val_x.npy', x_val) 
np.save('../MAT/val_y.npy', y_val) 

