#피쳐맵 바로 생성 후 분석위해서 csv로 일단

from pre_processing import *
import pandas as pd
import librosa


def input_data(file_name,direc):

    x, sr = librosa.load(direc + '/' + file_name )
    data = make_Feature([[x, sr]])
    data = pd.DataFrame(data)
    print(data)
    data.to_csv('output_{}.csv'.format(file_name), encoding='euc-kr')



input_data('음성 005_sd.m4a','C:/Users/wnduq/Downloads')

