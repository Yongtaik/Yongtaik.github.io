# AIX:DeepLearning Project
# Title: CNN-RNN 하이브리드 모델을 이용한 보이스 노이즈 제거 및 음질 향상

# Members & Roles
자원환경공학과 최용석 cys425922@gmail.com   
경영학과 구현태 kht999123@gmail.com   
경영학과 유영찬 reynoldred@hanyang.ac.kr   
전기공학과 최태환 middler@hanyang.ac.kr

# Youtube Link
https://


## Index
[1. Proposal](#1-proposal)\
[2. Datasets](#2-datasets)\
[3. Methodology](#3-methodology)\
[3. 1 사용 모델: CRNN이란?](#3-1-사용-모델-crnn이란)\
[3. 2 오디오 데이터 처리](#3-2-오디오-데이터-처리)\
[3. 3 오디오 데이터 처리 코드](#3-3-오디오-데이터-처리-코드)\
[3. 4 모델 생성](#3-4-모델-생성)\
[4. Evaluation & Analysis](#4-evaluation--analysis)\
[5. Conclusion](#5-conclusion)\
[6. Reference](#6-reference)


## 1. Proposal
### Motivation
최근 GPT-4o 공개 등으로 보여지듯, 인공지능 기술의 발전으로 음성 인식 시스템의 정확도와 효율성이 크게 향상되었습니다. 그러나 저음질 음성 데이터는 여전히 높은 오류율을 초래하여 음성 인식 시스템의 성능을 저해하고 있습니다. 가정에서도 산업현장에서도 음성 기반 시스템의 활용도가 증가함에 따라, 음질 개선의 필요성은 더욱 커지고 있습니다. 저음질 음성을 고음질로 변환하는 기술은 음성 데이터의 품질을 개선하여 음성 인식의 정확도를 높일 수 있습니다. 또한 기존에 사용할 수 없었던 저품질의 데이터를 활용가능하도록 변환시키면서 발생하는 양적인 기여 또한 무시할 수 없습니다. 결론적으로, 저음질 음성을 고음질로 변환하는 기술 개발은 음성 인식 시스템의 상용화와 효율적인 운영에 중요한 기여를 할 수 있습니다.


## 2. Datasets
잡음 제거 학습을 위해선 노이즈가 있는 음성과 노이즈가 없는 클린한 음성의 데이터가 쌍으로 필요합니다. 하지만 이러한 데이터 쌍을 구하기 쉽지 않기 때문에 두가지 음성을 합성하여 데이터 쌍을 만드는 방식을 사용했습니다.
> 데이터 합성 방식 참조[[1]](#1-httpsengineeringlinecorpcomkoblogvoice-waveform-arbitrary-signal-to-noise-ratio-python)

음성 데이터: CMU ARCTIC Databases의 음성파일(wav) 407개
> 음성 데이터 출처[[2]](#2-httpfestvoxorgcmu_arctic)

노이즈 데이터: DEMAND의 서로 다른 종류의 생활 소음 파일(wav) 5개
> 노이즈 데이터 출처[[3]](#3-httpszenodoorgrecords1227121w2wuvnj7tui)


<br>

### 데이터셋 상세 
음성 데이터는 5초 이내의 남자 목소리(영어) 203개와 여자 목소리(영어) 204개를 합쳐서 총 407개의 클린한로 사용,<br>
이를 train(300개), valid(99개), test(8개)로 분류했습니다. <br>
speech_synthesis.py를 통해 5분 길이의 5가지의 생활 소음(식당소리, 음악소리, 세탁기, 운동장소리, 공원소리)을 적용하고, 추가적으로 소음이 들어간 정도인 SNR(음성 대비 노이즈 비)값을 15, 20, 25 3가지로 설정하여 총 **6105**(407x5x3)개의 노이즈가 섞인 음성 데이터를 만들었습니다.<br>
데이터 셋의 분포는 아래와 같습니다.
<br>
<br>
train set : 4500개
<br>
valdation set : 1485개
<br>
test set : 120개
<br>
**total : 6105개**

<br>
<br>

### 데이터셋 합성 과정
<br>
<img src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/9f8182c1-37fa-46f2-9320-52cbc7326e13">

<br>
<br>
<br>

### SNR
노이즈 합성 비율을 결정하기 위한 음성과 노이즈의 대비값
<br>
<br>
<img width="240" alt="image" src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/aa161b58-f87b-4277-be50-0e9d57686681">
<br>
<br>
<br>
A<sub>signal</sub>와 A<sub>noise</sub>는 wav파일의 진폭의 제곱합의 RMS(평균 제곱근)값입니다.<br>
SNR 값을 입력하여 위의 식에서 도출된 값을 아래 식에 대입해 노이즈의 RMS를 구합니다. 
<br>
<br>
<img width="200" alt="스크린샷 2024-06-15 오전 4 26 12" src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/c52ef434-b406-4e44-9518-a9354d528e0b">
<br>
<br>
<br>
SNR값을 이용하여 구한 노이즈와 원본 노이즈의 비율을 아래와 같은 식으로 구하여 그 비율을 노이즈에 곱한 뒤, 음성과 더해서 노이즈가 있는 음성을 합성합니다. 이때, 음성 파일과 노이즈 파일의 길이가 다른 경우 더 긴 파일을 짧은 길이의 파일의 길이에 맞춥니다.
<br>
<br>
<img width="200" alt="스크린샷 2024-06-15 오전 4 39 07" src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/cf4bc267-ccdb-4457-89dd-cef538913a12">
<br>
<img width="700" alt="image" src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/b6114d0c-c5ec-4e81-93ff-5d1753567cfc">
<br>
<br>
<br>
<br>





## 3. Methodology
### 3. 1 사용 모델: CRNN이란? 
![CRNN_structure](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/be8a8361-7a76-4e1c-a980-2af27740b9f2)
<br>
<br>
CRNN(Convolutional Recurrent Neural Network)은 CNN(Convolutional Neural Network)과 RNN(Recurrent Neural Network)의 장점을 결합한 신경망의 한 유형입니다. CRNN은 먼저 CNN을 활용하여 이미지나 스펙트로그램과 같은 입력 데이터에서 공간적 특징을 계층적으로 추출합니다. 이렇게 추출된 특징을 RNN이 순차적으로 처리하여 시간적 종속성을 캡처함으로써, 시간에 따른 연속적인 정보를 효과적으로 다룰 수 있습니다. 이 아키텍처는 두가지 모델을 결합한만큼 지역적 특징 패턴과 장기적인 종속성을 모두 학습할 수 있다는 점에서 유리합니다. CRNN은 특히 음성 및 오디오 처리와 같은 응용 분야에서 사용되며, 입력 신호의 공간적 및 시간적 특성을 모델링하여 음성 향상과 같은 작업에서 음성의 명료성과 품질을 향상시키는 데 효과적입니다
<br>
저희가 참고한 논문에서는 이러한 모델을 주파수와 시간으로 표현된 2D 스펙트로그램 이미지를 활용하는 CNN 영역, 양방향 RNN 영역, 그리고 fully-connected 된 예측 영역의 세가지 단계로 표현했습니다.
> 이미지 및 알고리즘 개념 출처[[4]](#4-zhao-han-et-al-convolutional-recurrent-neural-networks-for-speech-enhancement-arxivorg-2-may-2018-httpsarxivorgabs180500579)

<br>
<br>
<br>

### 3. 2 오디오 데이터 처리

해당 모델은 오디오 데이터를 텐서 형태로 불러와야 합니다. 따라서 먼저 오디오를 시간적으로 작게 쪼개어 각각을 푸리에 변환한 2D 형태의 이미지로 변환합니다. 이 때 사용하는 기법을 **Short-time Fourier Transform (STFT)** 이라고 합니다.
<br>
<br>
오디오 데이터는 시간에 따라 Amplitude가 명시된 1차원 벡터입니다.
<br>
$Waveforms.shape = (1 ,Sample Rate (F_s) × Time (sec))$
<br>
<br>
이를 STFT를 거쳐서 변환시켜준 결과는 다음과 같습니다.
<br>
$Spectrogram.shape = (2 ,Frames,Frequency Bins)$
<br>
<br>
이때 첫글자의 '2'는 2차원임을 의미합니다. 이는 푸리에 변환을 거치면서 크기라고 할 수 있는 Amplitude Spectrogram과, 위상이라 할 수 있는 Phase Spectrogram 두 가지로 고유한 정보를 가지게 되기 때문입니다. 이는 마치 채널 수가 두 개인 이미지 텐서를 처리하는 것과 같은 것으로 볼 수 있습니다.
<br>
<br>
<br>
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/91a363a9-2e37-40e6-ae2a-ee56adc9c33d)
<br>
<br>
모델은 위의 두가지 스펙트로그램 중 Amplitude Spectrogram을 예측하도록 설계하는 것이 일반적입니다. 이유는 Phase Spectrogram을 그 자체로 처리하여 복원하는 것은 매우 직관적이지 못한 데이터를 주기 때문입니다. 따라서 저희는 모델에 Amplitude Spectrogram을 포워딩 시켜, 목소리에서 노이즈의 주파수 정보만 제거된 Amplitude Spectrogram을 얻는 것을 목표로 합니다. 그 이후 모델에 통과하기 전 얻어진 Phase Spectrogram을 사용하여 Inverse STFT를 통해 다시 오디오로 복원하도록 합니다.
<br>
<br>
<br>

![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/c3d9bbb6-29c1-44a9-b9d4-48e33794b61d)
<br>
<br>
파이썬의 모델에서 오디오를 텐서 형태로 다루기 위해서는 STFT 작업이 필수적입니다. 여기서 오디오의 샘플레이트를 F<sub>s</sub>라고 하고 FFT의 size를 N이라고 한다면 주파수 축의 해상도는 다음과 같습니다.
<br>
$Frequency Resolution(F_∆ )=F_s/N$
<br>
<br>
위의 F<sub>Δ</sub> 가 낮을수록 주파수를 더욱 정밀하게 표현할 수 있으며, 이는 곧 파이썬 환경에서 데이터의 세로축을 얼마나 촘촘하게 표현할 수 있는지를 의미합니다.
<br>
<br>
<br>
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/6473f2c4-010e-4382-979a-b7a26723ae23)
<br>
<br>
STFT를 거친 모델은 출력한 위와 같은 이미지 데이터를 출력합니다. 이러한 결과물을 처리하여 오디오의 노이즈를 제거하는 것이 저희의 다운스트림 태스크이기 때문에, 데이터에 대한 세부적인 이해가 중요합니다.
<br>
오른쪽의 ‘Low F<sub>Δ</sub>‘ 이미지 하단에서 두 개의 주파수가 강하게 존재하고 있음이 확인되나, 왼쪽의 ‘High F<sub>Δ</sub>‘ 이미지에선 그러한 모습을 확인할 수 없습니다. 저희는 모델 학습 과정에서 오른쪽과 같이 명확한 이미지를 사용하는 것이, 목소리와 노이즈를 주파수적으로 구분하는 데에 있어 유리함을 논문을 통해서 확인했습니다.[[5]](#5-tirronen-saska-et-al-the-effect-of-the-mfcc-frame-length-in-automatic-voice-pathology-detection-journal-of-voice-apr-2022-httpsdoiorg101016jjvoice202203021)
<br>
<br>
FFT 사이즈를 높이면 주파수 해상도는 증가하지만, 시간축의 해상도는 감소합니다. 그리고 시간 변화율이 잘 보이지 않는다는 것은, 이미지의 가로 해상도가 낮다는 것을 의미합니다. 저희 모델은 LSTM을 넣어 오디오의 시간적 특성도 충분히 고려하는 것을 추구하기 때문에 이는 바람직한 방향이 아닙니다. 따라서 FFT를 보정해줄 수 있는 **Hop_length**를 결정하여 시간적인 보정이 들어갈 수 있도록 해야 합니다.
<br>
<br>
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/4d29b703-9c08-4f89-a22f-2d271b6791b4)
<br>
<br>
위의 이미지는 Hop Length가 FFT Size의 절반일 때, 0.05초의 오디오에서 시간적인 변화를 2번만 반영하였으나, FFT Size의 1/6 일 때, 총 6번 변화에 대해 반영이 된 것을 알 수 있습니다.
<br>
저희의 데이터셋은 오디오 데이터의 샘플레이트(Fs)가 16000 이기 때문에, 이를 기준으로 위와 결과를 반영하여 FFT_size = 960, Hop_length = 160 으로 설정하였습니다. 

<br>
<br>
<br>

### 3. 3 오디오 데이터 처리 코드
오디오 (4초길이, 샘플레이트=16000) 데이터들을 불러와 Array 형태로 변환하는 코드입니다.
<br>
```python
import os
import librosa
import numpy as np
import torch
# 파일 경로 설정
label_folder = 'dataset/label'
mixed_folder = 'dataset/mixed'
batch_size = 8
sampling_rate = 16000
audio_length = 4 * sampling_rate  # 4초의 길이
# 파일 리스트 가져오기
label_files = sorted([os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.wav')])
mixed_files = sorted([os.path.join(mixed_folder, f) for f in os.listdir(mixed_folder) if f.endswith('.wav')])
# STFT 파라미터 설정
n_fft = 960
hop_length = 160
def load_audio(file_path, sr=sampling_rate):
    y, _ = librosa.load(file_path, sr=sr)
    y = librosa.util.fix_length(y, size=audio_length)  # 길이를 4초로 맞추기
    return y
def compute_stft(y, n_fft=n_fft, hop_length=hop_length):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    amp = np.abs(stft)
    phase = np.angle(stft)
    return amp, phase
# 배치 리스트 초기화
batches_mixed_amp = []
batches_clean_amp = []
batches_mixed_phase = []
batches_clean_phase = []
# 파일을 순차적으로 처리하여 배치로 나누기
for i in range(0, len(label_files), batch_size):
    mixed_amp_list = []
    mixed_phase_list = []
    clean_amp_list = []
    clean_phase_list = []
    
    for j in range(batch_size):
        if i + j >= len(label_files):
            break
            
        label_file = label_files[i + j]
        mixed_file = mixed_files[i + j]
        
        # 오디오 파일 로드
        clean_audio = load_audio(label_file)
        mixed_audio = load_audio(mixed_file)
        
        # STFT 계산
        mixed_amp, mixed_phase = compute_stft(mixed_audio)
        clean_amp, clean_phase = compute_stft(clean_audio)
        
        mixed_amp_list.append(mixed_amp)
        mixed_phase_list.append(mixed_phase)
        clean_amp_list.append(clean_amp)
        clean_phase_list.append(clean_phase)
    
    # 리스트를 넘파이 배열로 변환하고 형태 조정
    mixed_amp_batch = np.array(mixed_amp_list)
    mixed_phase_batch = np.array(mixed_phase_list)
    clean_amp_batch = np.array(clean_amp_list)
    clean_phase_batch = np.array(clean_phase_list)
    
    # 배치를 리스트에 추가
    batches_mixed_amp.append(mixed_amp_batch)
    batches_clean_amp.append(clean_amp_batch)
    batches_mixed_phase.append(mixed_phase_batch)
    batches_clean_phase.append(clean_phase_batch)

# 넘파이 배열 형태로 변환
batches_mixed_amp = np.array(batches_mixed_amp)
batches_clean_amp = np.array(batches_clean_amp)
batches_mixed_phase = np.array(batches_mixed_phase)
batches_clean_phase = np.array(batches_clean_phase)

# 출력 형태 확인
print(f"Mixed Amplitude Batch Shape: {batches_mixed_amp.shape}")  
print(f"Clean Amplitude Batch Shape: {batches_clean_amp.shape}")  
print(f"Mixed Phase Batch Shape: {batches_mixed_phase.shape}")    
print(f"Clean Phase Batch Shape: {batches_clean_phase.shape}")  

```
<br>

**출력 결과**
<br>

```
Mixed Amplitude Batch Shape: (20, 16, 481, 401)
Clean Amplitude Batch Shape: (20, 16, 481, 401)
Mixed Phase Batch Shape: (20, 16, 481, 401)
Clean Phase Batch Shape: (20, 16, 481, 401)
```

[세로 길이 481 x 가로 길이 401]의 이미지가 성공적으로 불러와진 것을 확인했습니다.
<br>
여기서 모델의 인풋으로 Mixed Amplitude Batch가 사용됩니다. 이에 대한 이유는 다음 모델 부분에서 설명하겠습니다.
<br>
<br>
<br>

### 3. 4 모델 생성
오디오의 지역적인 특성과 시계열적인 특성을 모두 고려할 수 있도록 CNN과 LSTM이 결합된 형태의 모델을 선정하였습니다. 이 때 모델이 매우 Deep 한 형태를 띄기에, Gradient Vanishing 문제를 해결할 필요가 있습니다. 또한 음성으로부터 노이즈의 제거는 **(음성+노이즈) – (노이즈)** 의 큰 형태를 띄기에, 잔차 학습이 매우 효과적일 것이라 판단하였습니다. 따라서 마치 U-net 구조처럼 Encoder와 Decoder 부분 사이에 Skip Connection이 있는 형태를 차용하고, 가운데에 LSTM을 배치함으로써, CNN을 통해 얻어진 피처맵의 시간적 특성을 고려하는 영역을 추가했습니다.
<br>
<br>
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/49a61f0e-7dd4-4aa7-b8e8-e6f5e4451260)

> 이미지 및 개념 출처 [[6]](#6-tan-ke-and-deliang-wang-a-convolutional-recurrent-neural-network-for-real-time-speech-enhancement-interspeech-2018-httpswwwsemanticscholarorgpapera-convolutional-recurrent-neural-network-for-speech-tan-wangd24d6db5beeab2b638dc0658e1510f633086b601)

<br>
참고한 논문에선 LSTM 레이어가 2개 사용되지만, 본 프로젝트에선 자원적 한계로 인해 1개의 레이어를 사용했습니다. 게다가 논문에 제시되었던 사이즈보다 더욱 선명한 상태의 오디오 스펙트로그램을 사용하기 때문에 LSTM은 제시된 것보다 큰 차원을 받아들여야 합니다. 따라서 미리 선정한 데이터의 형태에 맞게 수정하였고, 최종적으로 사용한 모델의 클래스와 포워딩 구문은 다음과 같습니다.

<br>
<br>

**모델 CRN 클래스 선언 :**
```python
import torch.nn as nn
import torch.nn.functional as F
import torch

class CRN(nn.Module):
    def __init__(self):
        super(CRN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=256)

        # LSTM
        self.LSTM1 = nn.LSTM(input_size=3584, hidden_size=3584, num_layers=1, batch_first=True)

        # Decoder
        self.convT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.bnT1 = nn.BatchNorm2d(num_features=128)
        self.convT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.bnT2 = nn.BatchNorm2d(num_features=64)
        self.convT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.bnT3 = nn.BatchNorm2d(num_features=32)
        self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0),
                                         output_padding=(0,1)) # 기존 frequency bins만큼 맞추기 위해 패딩
        self.bnT4 = nn.BatchNorm2d(num_features=16)
        self.convT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.bnT5 = nn.BatchNorm2d(num_features=1)

```
<br>

**포워딩 :**
```python
    def forward(self, x):
        
        
        p1d = (0,0,1,0) # 시간 차원을 보존하기 위해 제로패딩 배열
        print(f"input : {x.shape}")
        if (x.ndim ==3): 
            x.unsqueeze_(1)
        else :
            pass # [batch, num_channels(1), T, F] 
        print(f"unsqueezed input : {x.shape}")
        x_en = F.pad(x, p1d)
        x1 = F.elu(self.bn1(self.conv1(x_en)))
        print(f"Conv1 Output : {x1.shape}")
        
        x1_en = F.pad(x1, p1d)
        x2 = F.elu(self.bn2(self.conv2(x1_en)))
        print(f"Conv2 Output : {x2.shape}")
        
        x2_en = F.pad(x2, p1d)
        x3 = F.elu(self.bn3(self.conv3(x2_en)))
        print(f"Conv3 Output : {x3.shape}")
        
        x3_en = F.pad(x3, p1d)
        x4 = F.elu(self.bn4(self.conv4(x3_en)))
        print(f"Conv4 Output : {x4.shape}")
        
        x4_en = F.pad(x4, p1d)
        x5 = F.elu(self.bn5(self.conv5(x4_en)))
        print(f"Conv5 Output : {x5.shape}")
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        print(f"permuted Output : {out5.shape}")
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        print(f"reshaped Output for lstm Input : {out5.shape}")
        # lstm

        lstm, (hn, cn) = self.LSTM1(out5)
        # reshape
        output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
        print(f"lstm output : {output.shape}")
        # ConvTrans
        res = torch.cat((output, x5), 1)
        print(f"concat lstm : {res.shape}")
        res_en = F.pad(res, p1d)
        res1 = F.elu(self.bnT1(self.convT1(res_en)))
        print(f"deConv1 out : {res1.shape}")
        res1 = torch.cat((res1, x4), 1)
        print(f"concat deConv1 : {res1.shape}")
        res1_en = F.pad(res1, p1d)
        res2 = F.elu(self.bnT2(self.convT2(res1_en)))
        
        print(f"deConv2 out : {res2.shape}")
        res2 = torch.cat((res2, x3), 1)
        print(f"concat deConv2 : {res2.shape}")
        res2_en = F.pad(res2, p1d)
        res3 = F.elu(self.bnT3(self.convT3(res2_en)))
        print(f"deConv3 out : {res3.shape}")
        res3 = torch.cat((res3, x2), 1)
        print(f"concat deConv3 : {res3.shape}")
        
        res3_en = F.pad(res3, p1d)
        res4 = F.elu(self.bnT4(self.convT4(res3_en)))
        print(f"deConv4 out : {res4.shape}")
        res4 = torch.cat((res4, x1), 1)
        print(f"concat deConv4 : {res4.shape}")
        # (B, channel(1), T. F)
        
        res4_en = F.pad(res4, p1d)
        res5 = F.relu(self.bnT5(self.convT5(res4_en)))
        print(f"deConv5 out : {res5.shape}")
        print(f"Squeezed deConv5 out : {res5.squeeze().shape}")
        return res5.squeeze()

```
<br>
모델은 CNN에서 시간적 정보를 손실하지 않으면서 LSTM에 전송해줄 수 있도록 하며, LSTM은 CNN의 Encoder 부분에서 얻어진 Feature map 차원과 주파수 차원이 곱해진 reshaped input을 받아 처리할 수 있도록 합니다. 이 때 시간축의 개수가 보존되기에, LSTM은 피처맵들의 시계열적 특성만을 효과적으로 고려할 수 있습니다. Deconvolution 은 Encoder의 Convolution Network와 Skip Connection으로 연결되어 있어, 잔차를 학습할 수 있도록 합니다.
<br>
<br>
모델이 들어온 인풋을 받아들이고 학습하기 위해선 Loss와 Optimizer가 필요합니다.
<br>
Loss는 데이터셋의 음성만 있는 정답 라벨의 Amplitude Spectrogram과 Predicted Amplitude Spectrogram의 차이를 나타내도록 합니다. 여기서 저희는 노이즈 제거를 수행하는 모델이기에, 지워야 할 곳에 대한 [Batch Size, Frequency Bins, Number of Frames(T)] 사이즈의 바이너리 마스크를 만들고, 이를 Predicted Amplitude Spectrograms, 정답 Amplitude Spectrograms 두 곳 각각에 곱해주어 이 둘의 차이를 줄일 수 있도록 학습시킵니다.
<br>

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def loss_function(target, ipt):
        print("clean shape in loss.py : ", ipt.shape) # B x F x T
        print("enhanced shape in loss.py : ", target.shape) # B x F x T
        return torch.nn.functional.mse_loss(target, ipt)

```
<br>
Optimizer는 Adam을 사용하며, 모델을 GPU에 불러와 초기화합니다.
<br>

```python
import torch
import matplotlib.pyplot as plt
# 모델 초기화
model = CRN().to(device="cuda")

# 옵티마이저
optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.006,
        betas = [0.9, 0.999])

```
<br>

**모델 학습 코드:**
<br>

```python
# Put data to target device
device = "cuda"
epochs = 4
batch_mixed = torch.from_numpy(batches_mixed_amp).to(device)
batch_clean = torch.from_numpy(batches_clean_amp).to(device)

for epoch in range(epochs):
    model.train()
    for i in range(20):
        ## Training
        
        # 1. Forward pass
        prediction_amp_spt = model(batch_mixed[i].permute(0,2,1)) # 시간축과 주파수축을 전환
        
        # 2. Calculate loss/accuracy
        prediction_amp_spt = prediction_amp_spt.permute(0,2,1)
        loss = loss_function(prediction_amp_spt, batch_clean[i])
        # 3. Optimizer zero grad
        optimizer.zero_grad()
    
        # 4. Loss backwards
        loss.backward()
    
        # 5. Optimizer step
        optimizer.step()
    
    ### Testing
    model.eval()
    with torch.inference_mode():
        librosa.display.specshow(librosa.amplitude_to_db(prediction_amp_spt[0].detach().cpu().numpy(), ref=np.max), sr=16000, hop_length=hop_length, x_axis='time', y_axis='log')
        print(f"Epoch: {epoch} | Loss: {loss:.5f}")

```

<br>
<br>
<br>

## 4. Evaluation & Analysis
epoch가 한번 진행되었을 때 모델 아웃풋인 amplitude_spectrogram에서 샘플 1개를 불러와 정답 레이블과 비교하였습니다.
<br>
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/5bac101c-c09c-4a27-b4df-db9ce819441b)
<br>
<br>
Prediction 이미지를 볼 때, 노이즈로 추정되는 부분이 검정색으로 제거되었음을 확인할 수 있습니다. Ground Truth와 비교하였을 때, 음성을 과하게 제거한 부분이 존재하나 이는 에포크를 늘려 조금씩 보완해 나갈 수 있습니다.

<br>
<br>
해당 모델은 음성에 추가된 노이즈의 제거를 목표로 하며, 오디오 데이터는 실제로 들어보면서 얻는 직관이 적지 않기 때문에 최종적으로 사람이 들을 수 있는 결과물을 출력하는 것이 중요합니다. 
<br>
다음은 오디오 파일을 불러와 모델을 통과시키고 파일로 저장해주는 인퍼런스 코드입니다. 위에서 미리 불러온 트레이닝 텐서를 파라미터 학습없이 통과시켜 얻은 Amplitude Spectrogram과, 기존의 Phase Spectrogram을 이용해 ISTFT를 수행한 뒤, 들을 수 있는 Waveform 형태로 저장합니다. 
<br>

```python
import torch
import numpy as np
import librosa
import soundfile as sf
import os
def istft(magnitude, phase, n_fft=n_fft, hop_length=hop_length):
    complex_spectrum = magnitude * np.exp(1j * phase)
    y = librosa.istft(complex_spectrum, hop_length=hop_length)
    return y
batch_mixed = torch.from_numpy(batches_mixed_amp).to(device)
mixed_phase = batches_mixed_phase[0]

for i in range(1):
    model.eval()
    prediction_amp_spt = model(batch_mixed[i].permute(0,2,1)) # 시간축과 주파수축을 전환
    prediction_amp_spt = prediction_amp_spt.permute(0,2,1).cpu().detach().numpy()
    with torch.no_grad():
        for j in range(8):
            # 1. Forward pass
            predicted_audio = istft(prediction_amp_spt[j], mixed_phase[j])
            output_file = f'dataset/enhanced/predicted_audio_{j+1}.wav'
            sf.write(output_file, predicted_audio, samplerate=16000)
            print(f'Saved {output_file}')

```
<br>
<br>

**결과물**
<br>
clean10_with_ch01_snr15.mp4: 기존 잡음과 음성이 합쳐져 있는 원본 파일
<br>

https://github-production-user-asset-6210df.s3.amazonaws.com/168409733/340231610-9a381f2b-1080-49c5-a51b-af819ad32086.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240617T083614Z&X-Amz-Expires=300&X-Amz-Signature=fe994aeddc6b6ada6041854165b3bb818c7e55e30d41a1f7e2e5b7bc3f67b37b&X-Amz-SignedHeaders=host&actor_id=168409733&key_id=0&repo_id=793423595

<br>
<br>
predicted_audio_1.mp4: 모델을 통과한 결과물입니다
<br>

<video width="320" height="240" controls>
<source src="https://github-production-user-asset-6210df.s3.amazonaws.com/168409733/340231630-017aee8a-a4a9-4ef6-8e6c-1dd4cd968a34.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240617T083602Z&X-Amz-Expires=300&X-Amz-Signature=d715864bb6a641d6d6cce00409f3563e8bf328ecf25ba2a96e43314961f129a7&X-Amz-SignedHeaders=host&actor_id=168409733&key_id=0&repo_id=793423595" type="video/mp4">
</video>


<br>
<br>
<br>

Evaluation Metric으로는 PESQ를 사용합니다. PESQ 스코어는 -0.5 ~ 4.5 범위의 값을 출력하며, 두 가지 소리가 얼마나 서로 유사한지를 가리킵니다. 높은 값일수록 두 소리가 유사하다고 할 수 있습니다. 저희는 PESQ 스코어에서 Wideband PESQ를 사용합니다. test셋을 사용하여 Average PESQ 스코어를 출력합니다.
> PESQ 스코어 개념 및 알고리즘 출처[[7]](#7-ktword-정보통신기술용어-해설-httpwwwktwordcokrtestviewviewphpno2751)[[8]](#8-httpsgithubcomludlowspesq)



```python
import torch
import numpy as np
import librosa
from pesq import pesq
from pesq import PesqError
def evaluate_pesq(model, batch_mixed, batch_clean, device, sampling_rate=sampling_rate):
    model.eval()
    pesq_scores = []

    with torch.no_grad():
        for i in range(1):
            # Forward pass
            mixed_amp_spt = batch_mixed[i]  
            clean_amp_spt = batch_clean[i]  
            prediction_amp_spt = model(mixed_amp_spt.permute(0,2,1)).permute(0, 2, 1).cpu().numpy()
            mixed_audio = mixed_amp_spt.cpu().numpy()
            mixed_phase = batches_mixed_phase[i]
            for j in range(batch_size):
                predicted_audio = istft(prediction_amp_spt[j], mixed_phase[j])
                clean_audio = istft(clean_amp_spt[j].cpu().numpy(), batches_clean_phase[0][j])

                try:
                    pesq_score = pesq(sampling_rate, clean_audio, predicted_audio, mode='wb')
                    pesq_scores.append(pesq_score)
                except PesqError as e:
                    print(f"Error calculating PESQ score: {e}")
                    pesq_scores.append(None)

    return np.array(pesq_scores)
device = "cuda"
batch_mixed = torch.from_numpy(batches_mixed_amp_test).to(device)
batch_clean = torch.from_numpy(batches_clean_amp_test).to(device)
pesq_scores = evaluate_pesq(model, batch_mixed, batch_clean, device)
average_pesq = np.mean([score for score in pesq_scores if score is not None])

print(f"Average PESQ score: {average_pesq}")
```
<br>

![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/dbb49d0d-cfc6-489e-be84-cddca09bc081)
<br>

**최종 PESQ Score: 3.034**
<br>
<br>
<br>

## 5. Conclusion

**결론 및 한계**
<br>
출력된 아웃풋을 들어볼 때, 시각적으론 괜찮은 것 같아도 노이즈가 들리는 듯한 뉘앙스를 받는 경우가 많습니다. 최신 노이즈 제거 모델을 살펴보면, 저희처럼 Phase Spectrogram을 모델 인풋에서 획득하여 사용하는 것이 아닌 것들도 존재합니다. 저희의 모델은 Amplitude Spectrogram만을 예측하여 Predict 하므로 모델 학습이 매우 잘되어도 PESQ 스코어를 올릴 수 있는 한계가 명확히 존재합니다. 따라서 Phase Spectrogram을 Amplitude Spectrogram과 함께 예측하거나, 보정된 형태를 사용할 수 있는 방법을 쓰거나, Waveforms 그 자체에서 모델 예측을 하는 등의 시도를 해볼 수 있을 것입니다. 그것을 고려하고자 한다면, 모델의 파라미터수가 매우 높아질 확률이 크다 할 수 있습니다. 저희가 구현한 모델은 실시간과 같이 현실적인 상황을 가정했을 때, 인퍼런스 속도가 매우 빠르기 때문에 쉽게 적용해볼 수 있는 모델이란 점에서 의의가 있다고 생각합니다. 
모델을 학습하기 앞서 오디오의 전처리를 어떻게 할 것인가는 중요한 문제일 것입니다. 저희는 통상적인 FFT length와 Hop_length를 바로 사용하기 보단, 직접 data를 시각화하여 실제 모델이 어떻게 정보를 받아들일 수 있을지 판단해보고 전처리 조건을 바꾸어 적용시켰습니다. 샘플레이트 환경은 16000으로 주어졌습니다. Nyquist 이론에 따르면 샘플레이트가 16000일 때 표현할 수 있는 최대 주파수는 8000hz 까지입니다. 이는 사람의 음성이 담고 있는 정보를 전부 담을 수 있다고 말하긴 어렵습니다. 샘플레이트가 가청 주파수를 모두 표현할 수 있는 44100 등으로 훈련한다면 음악에 사용하고자 보컬 녹음 등에도 사용이 가능해질 것 입니다.
저희는 MSE LOSS 를 사용하였으나, 미분가능한 PESQ나 si sdr, LMS 등 다양한 로스가 존재합니다. MSE LOSS는 현재 음성 분야에선 한계가 지적되어 최신 모델에선 다른 Loss를 사용하는 추세인 것을 확인하였으며 성능적 지표가 비교적 떨어지는 것을 Hwang et al. 을 참고해 확인했습니다. Loss를 바꾸는 것이 청취적 평가인 정성적으로도, PESQ metric의 정량적으로도 우수해질 수 있으리라 판단했습니다.
모델의 아웃풋을 이미지로 도시할 때, 일반적으로 오른쪽 끝부분이 음성의 존재와 무관하게 지워지는 현상이 있다는 것을 파악했습니다.
 
왼쪽의 이미지는 모델을 통해 predict한 Amplitude Spectrogram입니다. 우측의 실제 ground truth의 3.5초 부근에서 엄연히 음성 신호가 존재함을 볼 수 있으나, 왼쪽의 이미지는 이를 제거한 모습입니다. 여기엔 다양한 문제가 있겠지만 깃허브의 speech enhancement 관련 여러 이슈를 찾아본 결과, 모델 외적으로는 데이터셋이 작고 쉽게 오버피팅 할 수 있는 구조일 때 음성을 제거하는 경향이 있다고 합니다. 저희는 굉장히 작은 데이터셋으로 테스트하였고, 음성과 노이즈가 다양한 조합을 갖고 있지 않습니다. 데이터셋의 한계를 극복하기 위해서 SpecAugment(Park, Daniel S., et al.) 등의 오디오 데이터 어그멘테이션 기법을 사용하는 것이 좋을 것으로 생각합니다. 
저희는 Validation 부분을 따로 만들지 않았는데, 위에서 서술한 이유입니다. 작은 데이터셋을 사용하기 때문에 트레이닝 에포크가 높지 않고 러닝 레이트 또한 낮을 때 좋은 성능을 내는 경향이 있었고, 저희가 사용했던 Loss에 대한 신뢰도가 낮기 때문입니다. 만약 SpecAugment를 사용하고 Loss 를 변경한다면 Validation을 만들어 validation에 대한 pesq나 loss를 기준으로 트레이닝을 중단하는 반복문을 통해 학습시키는 것이 바람직합니다. 
마지막으로 음성을 저희가 4초 길이로 모두 잘라서 사용했기에, 모델 또한 여기에 맞추어져 설계되었습니다. 하지만 파이토치의 LSTM은 인풋 시퀀스의 타임 스텝에 상관없이 적용이 가능합니다. 따라서 이 모델의 인퍼런스는 오디오의 길이가 달라지더라도 작동하기에 간편한 사용이란 장점이 있습니다. 쉽고 가볍게 사용할 수 있는 고성능 AI 모델이란 지향성을 잘 보여주고 있다고 생각합니다.



## 6. Reference
###### [[1]](#2-datasets) https://engineering.linecorp.com/ko/blog/voice-waveform-arbitrary-signal-to-noise-ratio-python
###### [[2]](#2-datasets) http://festvox.org/cmu_arctic/
###### [[3]](#2-datasets) https://zenodo.org/records/1227121#.W2wUVNj7TUI
###### [[4]](#3-methodology) Zhao, Han, et al. “Convolutional-Recurrent Neural Networks for Speech Enhancement.” arXiv.org, 2 May 2018, https://arxiv.org/abs/1805.00579
###### [[5]](#오디오-데이터-처리) Tirronen, Saska, et al. “The Effect of the MFCC Frame Length in Automatic Voice Pathology Detection.” Journal of Voice, Apr. 2022, https://doi.org/10.1016/j.jvoice.2022.03.021.
###### [[6]](#3-4-모델-생성) Tan, Ke and Deliang Wang. “A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement.” Interspeech (2018). https://www.semanticscholar.org/paper/A-Convolutional-Recurrent-Neural-Network-for-Speech-Tan-Wang/d24d6db5beeab2b638dc0658e1510f633086b601
###### [[7]](#4-evaluation--analysis) [KTword 정보통신기술용어 해설] http://www.ktword.co.kr/test/view/view.php?no=2751
###### [[8]](#4-evaluation--analysis) https://github.com/ludlows/PESQ



* Kumar, A., Florêncio, D., & Zhang, C. (2015). Linear Prediction Based Speech Enhancement without Delay. arXiv preprint arXiv:1507.05717. Retrieved from https://arxiv.org/abs/1507.05717
