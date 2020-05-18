# 목차

* [Introduction](#Introduction)
* [Data pre-processing](#Data-pre-processing)
	+ [Vocal extraction and VAD](#Vocal-extraction-and-VAD)



# Introduction
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/%EA%B7%B8%EB%A6%BC1.png)


# Data pre-processing

먼저 가수들의 노래에서  'WaveUNet' 신경망을 이용하여 곡에서 **반주를 제거**하여 보컬만을 남긴다. 이후 분리된 보컬에서 **MFCC를 추출**한다.  추출된 MFCC를 이용하여 **가수들을 clustering**하여 그 결과를 labeling에 이용한다.label을 y값으로 이용하여 supervised learning을 해 볼 수 있다.이때 **CNN model과 MLP model을 이용하여 성능을 비교**하여 볼 것이다.  
 
## Vocal extraction and VAD
Waveunet from : [https://github.com/f90/Wave-U-Net](https://github.com/f90/Wave-U-Net)
VAD from : [https://github.com/wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad)

사전 학습된 Waveunet 신경망을 이용하여 음원으로부터 Vocal만을 추출하였다.이후 이 음성에서 보컬이 등장하는 구간을 찾아주어 나머지 구간을 제거해 준다. 이를 VAD(Voice Active Detection)라고 한다. 
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/sam_fig.png)

![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/sam_fig_2.png)

## Feature extraction
### Mel-Frequency Cepstral Coefficients(MFCCs)
MFCC는인간의 청각 시스템을 모방한 변환 함수를 이용하여 **고음역대의 변화에 덜 민감하게** 필터링한다.


 ⑴ 우선 입력 신호를 일정한 간격의 frame으로 나눈 뒤, 
 ![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/1_202N86YExc2Y3JupOZZptQ.png)

 **⑵ 프레임 마다 Periodogram Spectral Estimate을 만든다**. 
>Periodogram 은 도메인을 frequency로 변환하여 각 frequency마다의 음압을 계산하여  각 frame마다의 배음구조를 확인 할 수 있다. 이 과정은 FFT(Fast Fourier Transform)을 이용하여 수행된다.

⑶ 이렇게 나온 결과를 **Power spectrum**이라 하고 이 Power spectrum에 **Mel Filter bank**를 적용한 후,

⑷ 각 	필터의 에너지를 더해준다. 

⑸ 구해진 Filter bank의 에너지에 **log변환**을 취해준다.

⑹  **DCT()를 적용**한 뒤, 나온 coefficient의  2~13 만 남기고 나머지는 버린다.

form : [MFCC tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#comment-3266294515)

```python
mfcc = librosa.feature.mfcc(x,sr=sr_x)
print(mfcc.shape)
plt.figure(figsize=(15,7))
librosa.display.specshow(mfcc,sr=sr_x, x_axis='time')
```
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/output_13_2.png)

# Labeling by unsupervised learning
 곡에서 얻어진  MFCC는 프레임 당 길이 20의 vector로 표현된다. 이를 전체 프레임에 대하여 **평균**을 매겨
 **곡 하나 당 길이 20의 vector로** 만들어준다. 그 후 **Auto Encoder**를 이용하여 이를 **2차원으로 축소** 시킨 뒤, K-means 알고리즘을 이용하여 **군집화** 시켜 결과를 **데이터의 라벨로 이용**할 것이다.     
## Dimension reduction by Auto Encoder
오토인코더는 manifold learning을 위해 주로 이용되며, 주로 
### Stacked Auto Encoder

### K-MEANs




![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/fig3.png)
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/fig4.png)

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTgyMzY5NDI3NCwxNTkyODc5Nzc4LDIwOD
k5NTIzNjAsLTU3MDY3MTUxNywtMTMwMjU0NDYwNSwtMTYwNTg3
MTc0NywtMTI3MjQzNjE5OSwzOTgwNDc1Nyw2MTgyNDYxMjMsLT
E0NDUyNzkyOTEsMTAxMzcyNjg0NSwtMTk3NjQxNTE5MSwtOTUw
MTE2MDU3LDIwOTAyMDAyMzUsLTM5OTcyMTk4MywzNjczMTIxOT
csLTc2MzUwODg5NiwxNjQyODc1Mzg5LDE5NzM1MjMyNjQsLTE3
NDYwODExMjRdfQ==
-->