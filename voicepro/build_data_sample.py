from pre_processing import *

song_list=["장범준-01-흔들리는 꽃들 속에서 네 샴푸향이 느껴진거야.mp3",
           '장범준-03-노래방에서.mp3',
           '버스커 버스커-04-벚꽃 엔딩.mp3',
           '03. 사랑한 후에.mp3',
           '01. HAPPY TOGETHER.mp3']
cut_list=[]
vocal_list=[]

for i in song_list:
    cut_list.append(i[:len(i)-4] + '.wav')
    vocal_list.append(i[:len(i)-4])
vocal_list.append('gift_mix')


input_lo = 'C:/Users/wnduq/Desktop/input_music'
output_lo = 'C:/Users/wnduq/Desktop/output_music'

#cut_music(song_list, input_lo, output_lo)
#voice_extraction(cut_list, input_lo, output_lo)
sam_list = sampling(vocal_list, output_lo)
data = MFCCs(sam_list)

import pandas as pd
data = pd.DataFrame(data)
data.index = ['장범준','장범준','장범준','박효신','박효신','박효신']
print(data)
data.to_csv('data.csv', encoding = 'euc-kr')






