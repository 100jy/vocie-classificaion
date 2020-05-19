import librosa
import librosa.display
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import *
from tensorflow.python.keras import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import *
from numpy import dot
from numpy.linalg import norm
import warnings
warnings.filterwarnings(action='ignore')

def run(file_path, db_path):

    test_x,sr = librosa.load(file_path)

    frame_length = 0.025
    frame_stride = 0.0125
    sr = 22000
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))
    mfcc = librosa.feature.mfcc(test_x[:22000 * 30], sr, n_mfcc=40, fmax=3000,
                                n_fft=input_nfft, hop_length=input_stride)
    mfcc_2 = librosa.feature.mfcc(test_x[:22000 * 30], sr, n_mfcc=20, fmax=3000,
                                  n_fft=input_nfft, hop_length=input_stride)

    x_mean = np.mean(mfcc_2, axis=-1).reshape(1, 20)

    num_rows = 200
    num_cols = 160
    num_cha = 3
    x_full = np.ravel(mfcc)[:96000].reshape(1, 96000)

    from sklearn.preprocessing import StandardScaler
    data1 = pd.read_pickle(db_path + '/for_scaling.pickle')
    scale = StandardScaler()
    scale.fit(data1)
    x_full = scale.transform(x_full)
    x_full = x_full.reshape(1, num_rows, num_cols, num_cha)

    data = pd.read_csv(db_path + '/data_male_full_30sec.csv', encoding='euc-kr', index_col='Unnamed: 0')
    scale2 = StandardScaler()
    scale2.fit(data.iloc[:, :20])
    x_mean = scale2.transform(x_mean)

    cluster = pd.read_csv(db_path + '/label_2.csv', encoding='euc-kr', index_col='Unnamed: 0')
    list_culster = ['a', 'b', 'c', 'd', 'e']



    backend.clear_session()

    # set_architecture
    VGG = VGG16(weights='imagenet', include_top=False, input_shape=(num_rows, num_cols, num_cha))
    VGG.trainable = False
    pre_model = Sequential()

    pre_model.add(VGG)
    pre_model.add(GlobalAveragePooling2D())
    pre_model.add(Dense(100))
    pre_model.add(Dropout(0.3))
    pre_model.add(Dense(5, activation='softmax'))
    adam = optimizers.Adam(lr=0.001)
    pre_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    # load_weight
    pre_model.load_weights(db_path + '/best_cnn_VGG.hdf5')
    predict_CNN = pre_model.predict_classes(x_full)
    print(cluster[cluster['1'] == list_culster[predict_CNN[0]]])


    def cosine_simularity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    db = pd.read_csv(db_path + '/mean_db.csv', index_col='Unnamed: 0')

    def get_singer(db, cluster, vector):
        arr = []
        for i in range(db.shape[0]):
            if db.iloc[i, :]['cluster'] == cluster:
                tmp = db.iloc[i, 1:21].reshape(20, )
                tmp2 = vector.reshape(20, )
                sim = cosine_simularity(tmp, tmp2)
                arr.append([sim, db.iloc[i, :]['0.1']])

        arr.sort(key=lambda x: x[0])
        return print('가장 유사한 가수 : ' + arr[-1][1] + '\n' + '유사도 : ' + str(round(arr[-1][0], 4)))

    return get_singer(db, list_culster[predict_CNN[0]], x_mean)


run('C:/Users/wnduq/Desktop/Python_code/voicepro/test/음성 008.m4a', 'C:/Users/wnduq/Desktop/Python_code/voicepro')