# AIX:DeepLearning Project
# Title: CRNN(CNN+RNN)을 활용한 음질 향상

# Members & Roles
자원환경공학과 최용석 cys425922@gmail.com   
경영학과 구현태 kht999123@gmail.com   
경영학과 유영찬 reynoldred@hanyang.ac.kr   
전기공학과 최태환 middler@hanyang.ac.kr


## Index
1. Proposal
2. Datasets
3. Methodology
4. Evaluation & Analysis
5. Conclusion
6. Reference

## 1.Proposal
### Motivation
최근 GPT-4o 공개 등으로 보여지듯, 인공지능 기술의 발전으로 음성 인식 시스템의 정확도와 효율성이 크게 향상되었습니다. 그러나 저음질 음성 데이터는 여전히 높은 오류율을 초래하여 음성 인식 시스템의 성능을 저해하고 있습니다. 가정에서도 산업현장에서도 음성 기반 시스템의 활용도가 증가함에 따라, 음질 개선의 필요성은 더욱 커지고 있습니다. 저음질 음성을 고음질로 변환하는 기술은 음성 데이터의 품질을 개선하여 음성 인식의 정확도를 높일 수 있습니다. 또한 기존에 사용할 수 없었던 저품질의 데이터를 활용가능하도록 변환시키면서 발생하는 양적인 기여 또한 무시할 수 없습니다. 결론적으로, 저음질 음성을 고음질로 변환하는 기술 개발은 음성 인식 시스템의 상용화와 효율적인 운영에 중요한 기여를 할 수 있습니다.

### 사용 모델: CRNN이란?
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/3f1b66b8-2e44-4ae7-aa53-77ce7b2f2143)
![image](https://github.com/Yongtaik/Yongtaik.github.io/assets/168409733/3f1b66b8-2e44-4ae7-aa53-77ce7b2f2143)

CRNN(Convolutional Recurrent Neural Network)은 CNN(Convolutional Neural Network)과 RNN(Recurrent Neural Network)의 장점을 결합한 신경망의 한 유형입니다. CRNN은 먼저 CNN을 활용하여 이미지나 스펙트로그램과 같은 입력 데이터에서 공간적 특징을 계층적으로 추출합니다. 이렇게 추출된 특징을 RNN이 순차적으로 처리하여 시간적 종속성을 캡처함으로써, 시간에 따른 연속적인 정보를 효과적으로 다룰 수 있습니다. 이 아키텍처는 두가지 모델을 결합한만큼 지역적 특징 패턴과 장기적인 종속성을 모두 학습할 수 있다는 점에서 유리합니다. CRNN은 특히 음성 및 오디오 처리와 같은 응용 분야에서 사용되며, 입력 신호의 공간적 및 시간적 특성을 모델링하여 음성 향상과 같은 작업에서 음성의 명료성과 품질을 향상시키는 데 효과적입니다


## 2.Datasets
### 데이터셋 상세
총 용량 540MB, 3초 내외의 짧은 영어 음성 데이터. clean_sound(고음질), mixed_sound(저음질) 각각 1132개의 음성 파일로 구성되어 있습니다. 저음질과 고음질 데이터는 1대1로 대응되기 때문에 저음질 데이터를 입력받은 모델의 출력물이 고음질 데이터와 얼마나 일치하는지 비교하여 훈련의 성과를 판단할 수 있습니다.

## 3. Methology


## 4.Evaluation & Analysis


## 5. Conclusion


## 6. Reference
* Kumar, A., Florêncio, D., & Zhang, C. (2015). Linear Prediction Based Speech Enhancement without Delay. arXiv preprint arXiv:1507.05717. Retrieved from https://arxiv.org/abs/1507.05717
