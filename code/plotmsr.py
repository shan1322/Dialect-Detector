import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
from tqdm import tqdm

path = "../MSR/gu-in-Train/Audios/"
audios = os.listdir(path)


def load_files_mfcc(loc, file):
    mfcct = []

    for i in tqdm(range(len(file))):
        data, rate = librosa.load(loc + file[i])
        mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=13)
        if mfcc.shape == (13):
            mfcct.append(mfcc)
    return np.array(mfcct)

def plot(mat):
    files=np.load(mat)
    print(files.shape)
    for i in range(len(files)):
        plt.plot(files[i])
        plt.ylim((-600, 250))
        plt.savefig("../plots gujju/" + str(i) + ".png", dpi=1500)

load_files_mfcc(path,audios)
plot("../Mat/gujju.npy")
