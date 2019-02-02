import json
import librosa
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load dataset
df = pd.read_csv('../speakers_all.csv')
path = "../recordings_new/"
filename = df['filename']
native = df['native_language']
native = list(set(native))
filename = np.array(filename)
# Get a dict of label and native
x, y = [], []
token = []
nat = df['native_language']
nat = np.array(nat)
for i in range(0, len(native)):
    token.append(i)
native = np.array(native)
ma = 0
for i in tqdm(range(len(filename))):
    try:
        fi = (path + filename[i] + ".wav")
        data, rate = librosa.load(fi)
        mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=13)
        for j in range(0, len(native)):
            if (nat[i] == native[j]):
                y.append(token[j])
                x.append(mfcc)
    except:
        print("error")
x_train = (x)
x_train_1 = []
ma = 0
for i in x_train:
    if len(i[0]) > ma:
        ma = len(i[0])
for i in x_train:
    temp3 = pad_sequences(i, ma, padding='post', value=1)
    temp4 = pad_sequences(i, ma, padding='pre', value=1)
    x_train_1.append(temp3)
    x_train_1.append(temp3)

x_train_1 = np.array(x_train_1)
print(x_train_1.shape)
new_y = []
for i in y:
    new_y.append(i)
    new_y.append(i)
y = np_utils.to_categorical(new_y, 214)
x_train, x_test, y_train, y_test = train_test_split(x_train_1, y, test_size=0.1, random_state=42)
x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
np.save('../MAT/train_x.npy', x_train)
np.save('../MAT/train_y.npy', y_train)
np.save('../MAT/test_x.npy', x_test)
np.save('../MAT/test_y.npy', y_test)

map = dict()
for i in range(len(native)):
    map[token[i]] = native[i]

with open('data.txt', 'w') as f:
    json.dump(map, f)
