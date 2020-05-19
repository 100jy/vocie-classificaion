# 목차

* [Introduction](#Introduction)
* [Data pre-processing](#Data-pre-processing)
	+ [Vocal extraction and VAD](#Vocal-extraction-and-VAD)
	+ [Feature extraction](#Feature-extraction)
		+ [Mel-Frequency Cepstral Coefficients(MFCCs)](#Mel-Frequency-Cepstral-Coefficients)
* [Labeling by unsupervised learning](#Labeling-by-unsupervised-learning) 
	+ [Dimension reduction by Auto Encoder](#Dimension-reduction-by-Auto-Encoder)
	+ [Stacked Auto Encoder](#K-MEANs)
* [Classification by Deep-Learning model](#Classification-by-Deep-Learning-model) 
	+ [MLP model](#MLP-model)
	+ [CNN model](#CNN-model)
+ [Input test ](#Input-test)
+ [Conclusion](#Conclusion) 
	



# Introduction
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/%EA%B7%B8%EB%A6%BC1.png)


# Data pre-processing

[전체 코드](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/pre_processing.py)

먼저 가수들의 노래에서  'WaveUNet' 신경망을 이용하여 곡에서 **반주를 제거**하여 보컬만을 남긴다. 이후 분리된 보컬에서 **MFCC를 추출**한다.  추출된 MFCC를 이용하여 **가수들을 clustering**하여 그 결과를 labeling에 이용한다.label을 y값으로 이용하여 supervised learning을 해 볼 수 있다.이때 **CNN model과 MLP model을 이용하여 성능을 비교**하여 볼 것이다.  
 
## Vocal extraction and VAD
Waveunet from : [https://github.com/f90/Wave-U-Net](https://github.com/f90/Wave-U-Net)

VAD from : [https://github.com/wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad)

사전 학습된 Waveunet 신경망을 이용하여 음원으로부터 Vocal만을 추출하였다.이후 이 음성에서 보컬이 등장하는 구간을 찾아주어 나머지 구간을 제거해 준다. 이를 VAD(Voice Active Detection)라고 한다. 
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/sam_fig.png)

![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/sam_fig_2.png)

## Feature extraction
### Mel-Frequency Cepstral Coefficients
MFCC는인간의 청각 시스템을 모방한 변환 함수를 이용하여 **고음역대의 변화에 덜 민감하게** 필터링한다.


 ⑴ 우선 입력 신호를 일정한 간격의 **frame으로 나눈 뒤,** 
 
 ![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/1_202N86YExc2Y3JupOZZptQ.png)

 **⑵ 프레임 마다 Periodogram Spectral Estimate을 만든다**. 
>Periodogram 은 도메인을 frequency로 변환하여 각 frequency마다의 음압을 계산하여  각 frame마다의 배음구조를 확인 할 수 있다. 이 과정은 FFT(Fast Fourier Transform)을 이용하여 수행된다.

⑶ 이렇게 나온 결과를 **Power spectrum**이라 하고 이 Power spectrum에 **Mel Filter bank**를 적용한 후,

⑷ 각 	필터의 에너지를 더해준다. 

⑸ 구해진 Filter bank의 에너지에 **log변환**을 취해준다.

⑹  **DCT(Discrete Cosine Transform)를 적용**한 뒤, 나온 coefficient의  2~13 만 남기고 나머지는 버린다.
> DCT는 DFT과 유사한 역할을 하며 처리 이후 에너지가 신호 성분이 낮은 주파수에 몰리게 되는 *에너지 집중 현상*이 나타나는 특징이 있다.  
[more info](https://users.cs.cf.ac.uk/Dave.Marshall/Multimedia/node231.html)

form : [MFCC tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#comment-3266294515)

```python
mfcc = librosa.feature.mfcc(x,sr=sr_x)
print(mfcc.shape)
plt.figure(figsize=(15,7))
librosa.display.specshow(mfcc,sr=sr_x, x_axis='time')
```
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/output_13_2.png)


# Labeling by unsupervised learning
 곡에서 얻어진  MFCC는 프레임 당 길이 20의 vector로 표현된다. 이를 전체 프레임에 대하여 **평균**을 매겨 **곡 하나 당 길이 20의 vector로** 만들어준다. 그 후 **Auto Encoder**를 이용하여 이를 **2차원으로 축소** 시킨 뒤, K-means 알고리즘을 이용하여 **군집화** 시켜 결과를 **데이터의 라벨로 이용**할 것이다.  
    
## Dimension reduction by Auto Encoder
오토인코더는 **manifold learning**을 위해 주로 이용되며, **Nonlinear dimensionality reduction**을 수행 할 수 있다.  먼저 데이터를 2차원으로 줄인 뒤, k-means 알고리즘을 이용하여  data point 간의 euclidean-dist를 기반으로 군집화 해 볼 것이다.

![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/maniford.png)

### Stacked Auto Encoder
다음과 같이 인코더와 디코더 부분이 **대칭**을 이루는 AE를  *Stacked Auto Encoder*라고 한다. 이때 인코더와 디코더는 가중치를 서로 공유한다.

<img src="https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/Structure-of-Stacked-Autoencoders.png" width="700" 
height="450"> 

[전체 코드](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/%EB%B6%84%EC%84%9D%EA%B3%BC%EC%A0%95/%EB%A7%8E%EC%9D%80_%EA%B0%80%EC%88%98%EC%97%90%EC%84%9C_%EA%B5%B0%EC%A7%91%ED%99%94.ipynb)
> 학습결과 : epoch : 999, **Train MSE : 0.09582**

*MSE : 0.09582*의 학습 결과는 인코딩 과정에서 **정보 손실이 매우 적다는 것**을 보여준다.  사전에 테스트 해본 **PCA의 결과였던 *0.3*정도**의 MSE 보다 훨씬 작은 값을 가지는 것을 볼 수 있다. 

### K-MEANs
2차원 축소 후 K-MEANs 알고리즘을 이용한 군집화 결과는 아래와 같다. 이 군집 결과를 이용하여 labeling을 해준 뒤, **신경망 모델을 이용하여 classification을** 해볼 것이다.

![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/cluster.png)

# Classification by Deep-Learning model

전처리를 거친 곡의 MFCC를 feature로 하여 label을 Classification 하는 신경망 모델들을 학습 시켜 볼 것이다.

[전체 코드](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/%EB%B6%84%EC%84%9D%EA%B3%BC%EC%A0%95/MLP%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EB%B6%84%EB%A5%98.ipynb)

## MLP model

**22명의 가수**의 **총 920곡**에 대해  **20차원  벡터**를 추출하여 분류해보았다. 

```python
#structure
backend.clear_session()
model = Sequential()
model.add(Dense(20, input_shape=(20,), activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(5,activation = 'softmax'))
#fit
adam = optimizers.Adam(lr = 0.01)
checkpointer = ModelCheckpoint(filepath='./save_models/best_MLP.hdf5')
model.compile(loss = 'categorical_crossentropy',optimizer = adam, metrics = ['accuracy'])
hist_mlp = model.fit(x_train,y_train,batch_size = 30, epochs = 50, validation_split = 0.1,
                                                         callbacks=[checkpointer], verbose=1)
```
> 결과
> Epoch 50/50
604/604 [==============================] - 0s 285us/step - loss: 0.4900 - acc: 0.8328 - val_loss: 0.7955 - val_acc: 0.6912
**testset 정확도 : 0.75**

>학습과정은 아래와 같다.
>
>
> <img src="https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/MLP_hist.png" width="700" 
height="450"> 

>테스트 셋 결과에 대한 confusion matrix는 아래와 같다.
>
>
><img src="https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/confusion_mat_MLP.png" width="700" 
height="450"> 

group5에 대해서 분류 결과가 매우 좋지 않다.  이는 group5의 데이터 수가 비교적 부족하며,  group4와 특성도 유사하기 때문으로 보인다.
> 그룹4와 그룹5 
> 
>![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/%EA%B7%B8%EB%A3%B94.png)
![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/%EA%B7%B8%EB%A3%B95.png)


## CNN model

이전의 모델과는 다르게 **22명의 가수**의 **총 1920곡**에 대해  **96000차원  벡터**를 추출하여 분류해보았다.  학습 이전에 (n, 200, 160, 3)의 **200x160의 3개의 channel을 가지는 형태**로 reshape 해주었다. 이후 VGG16 model을 이용하여 전이학습을 하여 모델을 학습 시켰다.
```python
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import *

backend.clear_session()

VGG = VGG16(weights='imagenet', include_top=False,input_shape=(num_rows,num_cols,num_cha))
VGG.trainable =False
pre_model = Sequential()

pre_model.add(VGG)
pre_model.add(GlobalAveragePooling2D())
pre_model.add(Dense(100,activation = 'relu'))
pre_model.add(Dropout(0.3))
pre_model.add(Dense(5,activation = 'softmax'))

epochs = 500
batch_size = 50

checkpointer = ModelCheckpoint(filepath='./best_cnn_VGG.hdf5')
adam = optimizers.Adam(lr = 0.001)
pre_model.compile(loss = 'categorical_crossentropy',metrics=['accuracy'],optimizer=adam)
hist = pre_model.fit(x_train,y_train,batch_size=batch_size,
                         epochs =epochs, validation_split=0.1, verbose=1, callbacks = [checkpointer])
```  
> 결과
> Epoch 500/500
1209/1209 [==============================] - 31s 25ms/step - loss: 0.0838 - acc: 0.9801 - val_loss: 0.9971 - val_acc: 0.7259
> **test_정확도 : 0.6927083333333334**

> 학습과정
> 
> 
> <img src="https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/CNN_hist.png" width="700" 
height="450"> 

>테스트 셋 결과에 대한 confusion matrix
>
>
><img src="https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/confusion_mat_CNN.png" width="700" 
height="450"> 

대부분의 그룹을  group4로 잘못 분류하는 경향이 있다. 이는 **group4의 데이터 수가** 다른 그룹의 데이터 수보다  **매우 많기 때문**인 것으로 보인다.

# Input test



새로운 입력에 대해 어느 군집으로 분류하는 지를 테스트 해보고 코사인 유사도를 기반으로 **가장 유사한 보컬을 찾아주었다.**

>CNN model을 이용한 분류 결과
>
>![enter image description here](https://github.com/100jy/vocie-classificaion/blob/master/voicepro/figures/%EA%B7%B8%EB%A3%B95.png ) 


이후, 해당 결과를 바탕으로 군집 내에서 가장 유사한 보컬을 찾아보면 

```python
from numpy import dot
from numpy.linalg import norm
def cosine_simularity(a,b):
    return dot(a,b)/(norm(a)*norm(b))

def get_singer(db, cluster, vector):
    arr = []
    for i in range(db.shape[0]):
        if db.iloc[i,:]['cluster'] == cluster:
            tmp = db.iloc[i,1:21].reshape(20,)
            tmp2 = vector.reshape(20,)
            sim = cosine_simularity(tmp,tmp2)
            arr.append([sim,db.iloc[i,:]['0.1']])

    arr.sort(key=lambda x : x[0])
    return print('가장 유사한 가수 : '+arr[-1][1]+'\n'+'유사도 : '+str(round(arr[-1][0],4)))
 
get_singer(db, list_culster[predict_CNN[0]], x_mean)
```

>**CNN model**로 찾은 가장 유사한 가수  
>
>**로이킴** (유사도 : 0.1437)

>**MLP model**로 찾은 가장 유사한 가수
>
>**김동률** (유사도 : -0.1139)

노래를 듣고 비교해보면 비교적 유사한 목소리를 찾은 것을 알 수 있다.

>김동률
>
>[soundcloud](https://soundcloud.com/yb-100/test_1/s-SHUlxKbz9nb)


>로이킴
>
>[soundcloud](https://soundcloud.com/yb-100/test2-1/s-wqZu1I7GB6i)

>실제 음성
>
>[soundcloud](https://soundcloud.com/yb-100/test3/s-JgaHYdTpbz3)





# Conclusion

**MFCC를 이용한 가수들의 분류**가 직관적으로 분류되는 기준과 비슷하게 보이며 이를 신경망으로 분류하는 것이 **납득을 할만한 결과**를 보여주었다. 이를 이용하여 가수들을 **목소리 별로 tagging**을 하는 것이 가능할 것으로 보인다.    허나 데이터가 부족하여 분류 정확도가 그리 높지는 않았다. 이후 데이터를 더 많이 모은다면 더욱 좋은 성능의 분류기를 학습 시킬 수 있을 것으로 보인다.   
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4NTM3NTg2OTEsLTEwMTQyMzg4MzcsMj
YzNDg0OTQ2LDEyMjUwMzk3MzYsMTE2MDQwOTY5NCwtODE5MzQy
OTc4LC0xNDQ2ODY2OTE1LDQyNTQ0MzI1NiwxNzY3Njg5MzcsLT
E0NTg2OTM5NjEsMTk4OTA5ODQ1NCwxMjIzNzQyODMzLC0xNDI1
NTM2NDI1LC0xMzAwMjc3NDI3LC0xOTg5OTIyNjYsMjA3NjA5Nj
M2MCw3MzE4ODIzMSwxMDA3MzU0MTg0LDEyMTY5OTM2MzUsNTE1
NjUwMzgxXX0=
-->