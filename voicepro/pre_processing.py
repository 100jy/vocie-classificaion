
#음원 앞에서 30초만 짜르기
def cut_music(song_list, input_location, output_location):

    import librosa
    sample_list = []
    for song_name in song_list:
        vocal_only = input_location + '/' +song_name
        x, sr = librosa.load(vocal_only)
        sample_list.append([x, sr])
        x = x[:662424]
        librosa.output.write_wav(output_location + '/{}.wav'.format(song_name[:len(song_name)-4]), x, sr)

# 보컬만 추출
def voice_extraction(song_list, input_location, output_location):
    import sys
    sys.path.append('C:/Users/wnduq/Desktop/Python_code/voicepro/WaveUnet')
    from Predict import ex
    for i in song_list:
        r = ex.run(named_configs=['cfg.full_44KHz'],
                   config_updates={'input_path': input_location + '/' + i,
                                   'model_path' : 'C:/Users/wnduq/Desktop/Python_code/voicepro/WaveUnet/checkpoints'
                                                  '/full_44KHz/full_44KHz-236118',
                                   'output_path': output_location})

# 보컬추출 음원에서 샘플림
def sampling(vocal_list, vocal_location):
    import librosa
    sample_list = []
    for song_name in vocal_list:
        vocal_only = vocal_location + '/' + '{}.wav_vocals.wav'.format(song_name)
        x, sr = librosa.load(vocal_only)
        sample_list.append([x, sr])
    return sample_list

# MFCC 적용하여 특징 추출
def MFCCs(sample_list):
    import librosa
    import numpy as np
    data = []
    for i in sample_list:
        mfcc_mean = []
        peaces = librosa.feature.mfcc(i[0], sr=i[1])
        for j in peaces:
            seg_mean = np.mean(j)
            mfcc_mean.append(seg_mean)
        data.append(mfcc_mean)
    return data

