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
잡음 제거 학습을 위해선 노이즈가 있는 음성과 노이즈가 없는 클린한 음성의 데이터가 쌍으로 필요합니다. 하지만 이러한 데이터 쌍을 구하기 쉽지 않기 때문에 두가지 음성을 합성하여 데이터 쌍을 만드는 방식을 사용했습니다.\
> 데이터 합성 방식 참조[[1]](#1-httpsengineeringlinecorpcomkoblogvoice-waveform-arbitrary-signal-to-noise-ratio-python)

음성 데이터는 CMU ARCTIC Databases의 영어로 된 남자와 여자 음성파일(wav) 407개를 사용하였고 
> 음성 데이터 출처[[2]](#2-httpfestvoxorgcmu_arctic)

노이즈 데이터는 DEMAND의 서로 다른 종류인 생활 소음의 파일(wav) 5개를 사용했습니다.
> 노이즈 데이터 출처[[3]](#3-httpszenodoorgrecords1227121w2wuvnj7tui)

추가적으로 소음이 들어간 정도인 SNR(음성 대비 노이즈 비)에 따라 3가지로 나누어 총 **6105**(407x5x3)개의 노이즈가 섞인 음성 데이터를 만들었습니다. 

**데이터셋 합성 과정**
</br>
</br>
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/9f8182c1-37fa-46f2-9320-52cbc7326e13)

</br>
</br>
</br>
</br>
### SNR

노이즈 합성 비율을 결정하기 위한 음성과 노이즈의 대비값</br>
<img width="381" alt="image" src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/aa161b58-f87b-4277-be50-0e9d57686681">
</br>
</br>
</br>
</br>
A<sub>signal</sub>와 A<sub>noise</sub>는 wav파일의 진폭의 제곱합의 평균의 제곱근을 나타낸 RMS(평균 제곱근)값입니다.
SNR 값을 입력하여 위의 식에서 도출된 아래의 식을 이용해 노이즈의 RMS를 구합니다. 
</br>
</br>
<img width="304" alt="스크린샷 2024-06-15 오전 4 26 12" src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/c52ef434-b406-4e44-9518-a9354d528e0b">
</br>
</br>
</br>
</br>
SNR값을 이용하여 구한 노이즈와 원본 노이즈의 비율을 아래와 같은 식으로 구하고 그 비율을 노이즈에 곱하고 음성과 더해서 노이즈가 있는 음성을 합성합니다. 음성 파일과 노이즈 파일의 길이가 다른 경우 더 긴 파일을 짧은 길이의 파일의 길이에 맞춥니다.
</br>
</br>
<img width="268" alt="스크린샷 2024-06-15 오전 4 39 07" src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/cf4bc267-ccdb-4457-89dd-cef538913a12">
</br>
<img width="733" alt="image" src="https://github.com/Yongtaik/Yongtaik.github.io/assets/77503751/b6114d0c-c5ec-4e81-93ff-5d1753567cfc">
</br>
</br>
</br>
</br>
### 데이터셋 상세 
CMU ARCTIC Databases의 5초 이내의 남자 영어 목소리 203개와 여자목소리 204개를 음성 데이터로 사용하여 407개의 클린한 음성 데이터를 train(300개), valid(99개), test(8개)로 나눠서 speech_synthesis.py에서 DEMAND의 5분 길이의 5가지의 생활 소음(식당소리, 음악소리, 세탁기, 운동장소리, 공원소리)을 SNR 값을 15, 20, 25으로 3가지 값으로 합성하여 데이터를 만들었습니다. 데이터 셋의 분포는 아래와 같습니다.
</br>
</br>
train set : 4500개
</br>
valdation set : 1485개
</br>
test set : 120개
</br>
-total : 6105개







## 3. Methodology
### 사용 모델: CRNN이란?
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/3f1b66b8-2e44-4ae7-aa53-77ce7b2f2143)   
CRNN(Convolutional Recurrent Neural Network)은 CNN(Convolutional Neural Network)과 RNN(Recurrent Neural Network)의 장점을 결합한 신경망의 한 유형입니다. CRNN은 먼저 CNN을 활용하여 이미지나 스펙트로그램과 같은 입력 데이터에서 공간적 특징을 계층적으로 추출합니다. 이렇게 추출된 특징을 RNN이 순차적으로 처리하여 시간적 종속성을 캡처함으로써, 시간에 따른 연속적인 정보를 효과적으로 다룰 수 있습니다. 이 아키텍처는 두가지 모델을 결합한만큼 지역적 특징 패턴과 장기적인 종속성을 모두 학습할 수 있다는 점에서 유리합니다. CRNN은 특히 음성 및 오디오 처리와 같은 응용 분야에서 사용되며, 입력 신호의 공간적 및 시간적 특성을 모델링하여 음성 향상과 같은 작업에서 음성의 명료성과 품질을 향상시키는 데 효과적입니다


## 4. Evaluation & Analysis


## 5. Conclusion


## 6. Reference
###### [[1]](#2-datasets) https://engineering.linecorp.com/ko/blog/voice-waveform-arbitrary-signal-to-noise-ratio-python
###### [[2]](#2-datasets) http://festvox.org/cmu_arctic/
###### [[3]](#2-datasets) https://zenodo.org/records/1227121#.W2wUVNj7TUI
* Kumar, A., Florêncio, D., & Zhang, C. (2015). Linear Prediction Based Speech Enhancement without Delay. arXiv preprint arXiv:1507.05717. Retrieved from https://arxiv.org/abs/1507.05717
