# 목차
[TOC]

# 개요
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

 **⑵ 프레임 마다 Periodogram Spectral Estimate을 만든다**. 
>Periodogram 은 도메인을 frequency로 변환하여 각 frequency마다의 음압을 계산하여  각 frame마다의 배음구조를 확인 할 수 있다. 이 과정은 FFT(Fast Fourier Transform)을 이용하여 수행된다.

⑶ 이렇게 나온 결과를 Power spectrum이라 하고 이 Power spectrum에 **Mel Filter bank**를 적용한다

⑷ 구해진 Filter bank의 에너지에 **log변환**을 취해준다.

⑸ **DCT()를 적용**한 뒤, 나온 coefficient의  2~13 만 남기고 나머지는 버린다.





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
eyJoaXN0b3J5IjpbLTEyNzI0MzYxOTksMzk4MDQ3NTcsNjE4Mj
Q2MTIzLC0xNDQ1Mjc5MjkxLDEwMTM3MjY4NDUsLTE5NzY0MTUx
OTEsLTk1MDExNjA1NywyMDkwMjAwMjM1LC0zOTk3MjE5ODMsMz
Y3MzEyMTk3LC03NjM1MDg4OTYsMTY0Mjg3NTM4OSwxOTczNTIz
MjY0LC0xNzQ2MDgxMTI0LC0xNDgyOTA2NDQsOTY0NjA4OTUyLD
k2NDYwODk1Miw5NjY2NTcyMDYsLTgzNDk0MjEyOSwtMTk0MjIy
MjY3OV19
-->