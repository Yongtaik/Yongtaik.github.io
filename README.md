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
### 사용 모델: CRNN이란? 
![CRNN_structure](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/be8a8361-7a76-4e1c-a980-2af27740b9f2)
<br>
<br>
CRNN(Convolutional Recurrent Neural Network)은 CNN(Convolutional Neural Network)과 RNN(Recurrent Neural Network)의 장점을 결합한 신경망의 한 유형입니다. CRNN은 먼저 CNN을 활용하여 이미지나 스펙트로그램과 같은 입력 데이터에서 공간적 특징을 계층적으로 추출합니다. 이렇게 추출된 특징을 RNN이 순차적으로 처리하여 시간적 종속성을 캡처함으로써, 시간에 따른 연속적인 정보를 효과적으로 다룰 수 있습니다. 이 아키텍처는 두가지 모델을 결합한만큼 지역적 특징 패턴과 장기적인 종속성을 모두 학습할 수 있다는 점에서 유리합니다. CRNN은 특히 음성 및 오디오 처리와 같은 응용 분야에서 사용되며, 입력 신호의 공간적 및 시간적 특성을 모델링하여 음성 향상과 같은 작업에서 음성의 명료성과 품질을 향상시키는 데 효과적입니다
<br>
저희가 참고한 논문에서는 이러한 모델을 주파수와 시간으로 표현된 2D 스펙트로그램 이미지를 활용하는 CNN 영역, 양방향 RNN 영역, 그리고 fully-connected 된 예측 영역의 세가지 단계로 표현했습니다.
> 이미지 및 알고리즘 개념 출처[[4]](#4-zhao-han-et-al-convolutional-recurrent-neural-networks-for-speech-enhancement-arxivorg-2-may-2018-httpsarxivorgabs180500579)

<br>
<br>

### 오디오 데이터 처리

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
오른쪽의 ‘Low FΔ‘ 이미지 하단에서 두 개의 주파수가 강하게 존재하고 있음이 확인되나, 왼쪽의 ‘High FΔ‘ 이미지에선 그러한 모습을 확인할 수 없습니다. 저희는 모델 학습 과정에서 오른쪽과 같이 명확한 이미지를 사용하는 것이, 목소리와 노이즈를 주파수적으로 구분하는 데에 있어 유리함을 논문을 통해서 확인했습니다.


<br>
<br>
<br>
<br>

## 4. Evaluation & Analysis


## 5. Conclusion


## 6. Reference
###### [[1]](#2-datasets) https://engineering.linecorp.com/ko/blog/voice-waveform-arbitrary-signal-to-noise-ratio-python
###### [[2]](#2-datasets) http://festvox.org/cmu_arctic/
###### [[3]](#2-datasets) https://zenodo.org/records/1227121#.W2wUVNj7TUI
###### [[4]](#3-methodology) Zhao, Han, et al. “Convolutional-Recurrent Neural Networks for Speech Enhancement.” arXiv.org, 2 May 2018, https://arxiv.org/abs/1805.00579
###### [[5]](#오디오-데이터-처리) Tirronen, Saska, et al. “The Effect of the MFCC Frame Length in Automatic Voice Pathology Detection.” Journal of Voice, Apr. 2022, https://doi.org/10.1016/j.jvoice.2022.03.021.
* Kumar, A., Florêncio, D., & Zhang, C. (2015). Linear Prediction Based Speech Enhancement without Delay. arXiv preprint arXiv:1507.05717. Retrieved from https://arxiv.org/abs/1507.05717
