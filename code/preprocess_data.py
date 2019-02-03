import json

import librosa
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm

df = pd.read_csv('../speakers_all.csv')
location = "../recordings_new/"
file_name = df['filename']
native = df['native_language']


# print(pd.factorize(native)[0])


# dict(enumerate(df['x'].cat.categories))


class PreProcessing:
    def __init__(self):
        self.features = []
        self.labels = []
        self.tokes = []
        self.padded_features = []
        self.double_label = []

    def training_pre_process(self, file_names, path, native_language):
        """
        :param file_names: list of files
        :param path: location of files
        :param native_language: native languages
        :return: train features and labels
        """
        native_language_token = pd.factorize(native_language)[0]
        for counter in tqdm(range(len(file_names))):
            try:
                data, rate = librosa.load(path + file_names[counter] + ".wav")
                mfcc = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=13)
                self.features.append(mfcc)
                self.labels.append(native_language_token[counter])
            except:
                print("error loading file")
        self.labels = np_utils.to_categorical(self.labels, 214)
        print(self.labels)
        return self.features, self.labels

    def pad_sequences(self):
        max_length = 0
        for sequence in self.features:
            if len(sequence) > max_length:
                max_length = len(sequence)
        for sequence in self.features:
            black_padded_post = pad_sequences(sequence, max_length, padding='post', value=1)
            black_padded_pre = pad_sequences(sequence, max_length, padding='pre', value=1)
            self.padded_features.append(black_padded_post)
            self.padded_features.append(black_padded_pre)
        for labels in self.labels:
            self.double_label.append(labels)
            self.double_label.append(labels)
        return self.double_label, self.padded_features

    def train_test_split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.padded_features, self.double_label, test_size=0.1,
                                                            random_state=42)
        return np.save('../MAT/train_x.npy', x_train), np.save('../MAT/train_y.npy', y_train), np.save(
            '../MAT/test_x.npy', x_test), np.save('../MAT/test_y.npy', y_test)

    @staticmethod
    def dictionary_languages(native_languages):
        with open('../map/data.txt', 'w') as f:
            json.dump(dict(enumerate(sorted(set(native_languages)))), f)


obj = PreProcessing()
obj.training_pre_process(file_name, location, native)
obj.pad_sequences()
obj.train_test_split()
obj.dictionary_languages(native)
