from pre_processing import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def build_data():

    #song_new에 추가된 것만 추가
    song_list = get_new()
    print(song_list)


    cut_list = []
    vocal_list = []

    for i in song_list:
        cut_list.append(i[:len(i) - 4] + '.wav')
        vocal_list.append(i[:len(i) - 4])

    input_lo = 'C:/Users/wnduq/Desktop/input_music/sep_by_singer'
    output_lo = 'C:/Users/wnduq/Desktop/output_music'

    #cut_music(song_list, input_lo, output_lo)
    #voice_extraction(cut_list, output_lo, output_lo)
    #VAD(cut_list, output_lo, output_lo)
    sam_list = sampling(vocal_list, output_lo)
    data = make_Feature(sam_list)

    data = pd.DataFrame(data)
    data.index = vocal_list
    #data = mix_data(data)
    print(data)
    #data.to_csv('data_10sec.csv', encoding='euc-kr')
    data.to_csv('data.csv', encoding='euc-kr')


if __name__ == '__main__':
    build_data()

