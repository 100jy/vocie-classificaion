# 개요
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/%EA%B7%B8%EB%A6%BC1.png)

# Data pre-processing

먼저 가수들의 노래에서  'WaveUNet' 신경망을 이용하여 곡에서 **반주를 제거**하여 보컬만을 남긴다. 이후 분리된 보컬에서 **MFCC를 추출**한다.  추출된 MFCC를 이용하여 **가수들을 clustering**하여 그 결과를 labeling에 이용한다.label을 y값으로 이용하여 supervised learning을 해 볼 수 있다.이때 **CNN model과 MLP model을 이용하여 성능을 비교**하여 볼 것이다.  
 
## Vocal extraction and VAD
from : [https://github.com/f90/Wave-U-Net](https://github.com/f90/Wave-U-Net)

사전 학습된 Waveunet 신경망을 이용하여 음원으로부터 Vocal만을 추출하였다.이후 이 음성에서 보컬이 등장하는 구간을 찾아주어 나머지 구간을 제거해 준다. 이를 VAD(Voice Active Detection)라고 한다. 

## Sampling
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
%matplotlib inline 
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr_x)
plt.title('vocal_only')
```

![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/output_8_1.png)


## VAD
[webrtcvad](https://github.com/wiseman/py-webrtcvad)를 이용하여 반주가 제거 된 음원에서 보컬이 등장하는 구간 만을 추출하여 준다.
 

## Feature extraction
### Mel-Frequency Cepstral Coefficients(MFCCs)
10~20개의 스펙트럼의 subset, 인간의 청각 시스템을 모델링해서 고음역대의 변화에 덜 민감하게 필터링

```python
mfcc = librosa.feature.mfcc(x,sr=sr_x)
print(mfcc.shape)
plt.figure(figsize=(15,7))
librosa.display.specshow(mfcc,sr=sr_x, x_axis='time')
```

![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/output_13_2.png)

## Dimension reduction by Auto Encoder
 Auto Encoder를 이용하여 변수가 5만 정도 되는 고차원 데이터를 400개 가량의 변수로 축소 시킨다. 
## Labeling by unsupervised learning
### K-MEANs
추출된 특징을 기반으로 k - means 알고리즘을 통해 군집화 한 뒤, 해당 결과를 정답 Label으로 이용.



![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/fig3.png)
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/fig4.png)

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTEwOTIyMTU3OCwtNzYzNTA4ODk2LDE2ND
I4NzUzODksMTk3MzUyMzI2NCwtMTc0NjA4MTEyNCwtMTQ4Mjkw
NjQ0LDk2NDYwODk1Miw5NjQ2MDg5NTIsOTY2NjU3MjA2LC04Mz
Q5NDIxMjksLTE5NDIyMjI2NzksLTMxMDI3OTI1LDM2MDI1MTcx
MF19
-->