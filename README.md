# 개요
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/%EB%8C%80%EB%9E%B5%EC%A0%81%EC%9D%B8%20%EA%B5%AC%EC%A1%B0.png)

# Data pre-processing

```mermaid
graph LR
A[Full song] -- WaveUNet --> B(Vocal only)
B -- Extractiong Feature by MFCCs  --> C[Feature of song]
머메이드 지원안한다슈발럼들...
```

 'WaveUNet' 신경망을 이용하여 곡에서 반주를 제거한다. 이후 MFCCs 모델을 이용하여 특징을 추출하고 이를 이용하여 가수들을 분류한다. 이후 해당 데이터를 이용하여 학습된 분류기로 입력값에 대해 분류한다. 
## Vocal extraction by Waveunet
Vocal extracion from signal datas by using
Waveunet, cnn network designed for separating vocal from song

```python
from Predict import ex
song_list=['gift_mix.mp3']
local_path=['C:/Users/wnduq/Desktop/input_music/{}','C:/Users/wnduq/Desktop/output_music/{}']
for i in song_list:
    r = ex.run(named_configs=['cfg.full_44KHz'],
              config_updates={'input_path': local_path[0].format(i),
                             'output_path' : local_path[1].format(i[:len(i)-3])})
```

## sampling
sampling from continuous signal to create discrete array data. 
to
do this, choose the sampling tate, that determines how tight intervals between
samples are

```python
import librosa
local_path=['C:/Users/wnduq/Desktop/input_music/{}','C:/Users/wnduq/Desktop/output_music/{}']
vocal_only = local_path[1].format('gift_mix/gift_mix.mp3_vocals.wav')
mixed = local_path[0].format('gift_mix.mp3')
x , sr_x = librosa.load(vocal_only) #default sr is 22KHZ
y, sr_y = librosa.load(mixed)
print(x.shape, sr_x)
print(y.shape, sr_y)
```

```python
#cheak audio
import IPython.display as ipd
ipd.Audio(vocal_only)
```

```python
ipd.Audio(mixed)
```

```python
%matplotlib inline 
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr_x)
plt.title('vocal_only')
```

```python
%matplotlib inline 
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr_y)
plt.title('mixed')
```


## feature extraction
### Mel-Frequency Cepstral Coefficients(MFCCs)
10~20개의
스펙트럼의 subset, 인간의 청각 시스템을 모델링해서 고음역대의 변화에 덜 민감하게 필터링

```python
mfcc = librosa.feature.mfcc(x,sr=sr_x)
print(mfcc.shape)
plt.figure(figsize=(15,7))
librosa.display.specshow(mfcc,sr=sr_x, x_axis='time')
```

```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI2NzczNzkyMCwtMzEwMjc5MjUsMzYwMj
UxNzEwXX0=
-->