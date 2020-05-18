#추가된 파일 리스트
#같은 가수 음악 3개씩해서 평균 매기기
def get_list(input_path):
    import glob
    path = input_path + '/*'
    file_gen = glob.glob(path)
    file_list = [x for x in file_gen]
    return file_list

def get_new():
    import pandas as pd
    import numpy as np
    singers = pd.read_csv('./singers.csv', encoding='euc-kr')
    singers_new = pd.read_csv('./singers_new.csv', encoding='euc-kr')

    arr = np.array(singers_new['0'])
    arr2 = np.array(singers['0'])

    new = np.setdiff1d(arr, arr2)

    return new

# 음원 중간에서 30초만 짜르기
def cut_music(song_list, input_location, output_location):
    import librosa
    sample_list = []
    for song_name in song_list:
        vocal_only = input_location + '/' + song_name
        x, sr = librosa.load(vocal_only)
        sample_list.append([x, sr])
        #x = x[662424:662424*2]
        #전구간 다활용
        librosa.output.write_wav(output_location + '/{}.wav'.format(song_name[:len(song_name) - 4]), x, sr)



# 보컬만 추출
def voice_extraction(song_list, input_location, output_location):
    import sys
    sys.path.append('C:/Users/wnduq/Desktop/Python_code/voicepro/WaveUnet')
    from Predict import ex
    import os
    present = os.listdir(output_location)
    for i in song_list:
        tmp = i + '_vocals.wav'
        if tmp not in present:
            r = ex.run(named_configs=['cfg.full_44KHz'],
                       config_updates={'input_path': input_location + '/' + i,
                                       'model_path': 'C:/Users/wnduq/Desktop/Python_code/voicepro/WaveUnet/checkpoints'
                                                     '/full_44KHz/full_44KHz-236118',
                                       'output_path': output_location})


#16비트변환 후 VAD
def VAD(cut_list, input_lo):
    import sys
    sys.path.append('C:/Users/wnduq/Desktop/Python_code/voicepro/py-webrtcvad-master')
    from VAD_tool import main
    import librosa
    import soundfile as sf
    import os
    present = os.listdir('C:/Users/wnduq/Desktop/output_music')
    for i in cut_list:
        tmp = i + '_vocals_vad.wav'
        if tmp not in present:
            path = input_lo
            temp_x, sr = librosa.load('C:/Users/wnduq/Desktop/output_music/{}_vocals.wav'.format(i)
                                      , sr=32000)
            sf.write('C:/Users/wnduq/Desktop/output_music/{}_vocals.wav'.format(i), temp_x, sr,
                     format='WAV', endian='LITTLE', subtype='PCM_16')
            main([3, path + '/{}_vocals.wav'.format(i)])#숫자로 강도조절

# 보컬추출 음원에서 샘플림
def sampling(vocal_list, vocal_location):
    import librosa
    sample_list = []
    for song_name in vocal_list:
        vocal_only = vocal_location + '/' + '{}.wav_vocals_vad.wav'.format(song_name)
        x, sr = librosa.load(vocal_only)

        x_1 = x[0:30*22000] #1분
        x_2 = x[22000*2:22000*32]
        x_3 = x[22000*4:22000*34]
        x_4 = x[22000*6:22000*36]
        x_5 = x[22000 * 8:22000 * 38]
        x_6 = x[22000 * 10:22000 * 40]
        x_7 = x[22000 * 12:22000 * 42]
        x_8 = x[22000 * 14:22000 * 44]

        for i in [x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8]:
            sample_list.append([i, sr])

    return sample_list



# MFCC 적용하여 특징 추출
def make_Feature(sample_list):
    import librosa
    import numpy as np
    data = []
    frame_length = 0.025
    frame_stride = 0.0125
    sr = 22000
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))

    for i in sample_list:
        mfcc_mean = []
        # 수정 좀 해야될듯...
        if len(i[0]) != 0:
            mfcc = librosa.feature.mfcc(i[0], sr=i[1], n_mfcc=40, fmax=3000, n_fft=input_nfft
                                                , hop_length=input_stride)
            #mfccsscaled = np.mean(mfcc.T, axis=0)
            flat = np.ravel(mfcc)
        #집계하지 않은 full data 이용


        '''
        rmse = librosa.feature.rmse(y=i[0])
        chroma_stft = librosa.feature.chroma_stft(y=i[0], sr=i[1])
        spec_cent = librosa.feature.spectral_centroid(y=i[0], sr=i[1])
        spec_bw = librosa.feature.spectral_bandwidth(y=i[0], sr=i[1])
        rolloff = librosa.feature.spectral_rolloff(y=i[0], sr=i[1])
        zcr = librosa.feature.zero_crossing_rate(i[0])
        
        list_feature = [rmse,chroma_stft,spec_bw,spec_cent,rolloff,zcr]
        #list_feature = list(map(lambda x : np.mean(x), list_feature))
        list_feature = list_feature + mfcc_mean
        '''

        #data.append(mfccsscaled)
        #data.append(mfcc)
        data.append(flat)

    return data

###수정해야함....(열이름 통일하고 합쳐야함..)
def mix_data(new_data):
    import pandas as pd
    data_pre = pd.read_csv('./data.csv', encoding='euc-kr',index_col=0)
    con = data_pre.append(new_data)
    return con