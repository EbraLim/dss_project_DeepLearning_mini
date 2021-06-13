[Deep Learning mini project]\
OX classification with hand-written images
==
## 0. 목차
```
1. 소개
2. 진행 프로세스
3. 프로세스별 상세설명
4. 오분류 데이터 확인 및 원인 추정
5. 추후 보완점
6. 멤버 & 수행업무
```
----
## 1. 소개
### 1) 주제
* 손글씨로 생성한 O, X 이미지 데이터를 분류하는 딥러닝 모델 만들기
### 2) 목표
* 적합한 데이터 전처리 수행을 통한 학습데이터 품질 및 모델 성능 향상 (accuracy 95% 이상)
* 전처리 방법 모듈화  
### 3) 결과
* 최초 test accuracy: 76.6% ~ 84.4%  
* 최종 test accuracy: 98.4%
----

## 2. 진행 프로세스
<img width="274" alt="스크린샷 2021-05-21 오후 12 07 06" src="https://user-images.githubusercontent.com/78459305/119076168-11ceeb00-ba2d-11eb-9e8b-70f441f8cfca.png">

* 총 3번의 iteration 수행
      
----

## 3. 프로세스별 상세설명
### 1) 데이터 수집
* A4 용지를 64등분한 크기로 O,X 각각 320개씩 제작 후 직접 촬영
* 데이터 복잡성을 높이기 위해 선 굵기가 다른 5개의 펜 사용, 필적 학습 방지를 위해 양손으로 번갈아가며 제작
<img width="753" alt="스크린샷 2021-05-22 오전 10 34 30" src="https://user-images.githubusercontent.com/78459305/119210805-4c965900-bae9-11eb-9efa-9a1c24462005.png">

* raw data에 대해 크게 3가지 문제점이 있었으며, 아래와 같이 해결하고자 하였음  
  - 다양한 명도를 가진 컬러 이미지이며, 종이와 책상의 경게선이나 그림자 등 노이즈 존재  
    → 이진화 (binarization)를 통해 처리  
  - 펜 종류에 따라 매우 얇고 흐릿한 데이터 존재  
    → dilate와 erode의 적절한 조합을 통해 선이 두꺼워지도록 변환  
  - 이미지 내 O,X의 위치가 다양하게 분산되어 있음 (중앙, 구석)  
    → O,X의 ROI (Region Of Interest: 관심영역)만을 crop한 후 검은색 mask를 적용하여 이미지 중앙으로 위치 통일  

### 2) 전처리 (1차)
* 이미지 이진화(binarization) (threshold=50)  
<img width="753" alt="스크린샷 2021-05-22 오전 10 35 27" src="https://user-images.githubusercontent.com/78459305/119210823-6f287200-bae9-11eb-8642-927d28b66df4.png">  

* resize (원본 2160 x 2880 → 360 x 480 으로 축소)  
* 이미지 정규화 (0 ~ 255 사이의 pixel값을 0 ~ 1 사이의 값으로 변환)  
* O, X 데이터 통합 및 train/test 데이터셋 분리 (test size = 0.2)  

### 3) 모델링
* ANN과 CNN, 총 두가지 모델 사용
#### (1) ANN
* 입력층, 1개의 은닉층 (node 1000개, 'ReLU'), 출력층 (node 2개, 'Softmax')으로 구성  
* optimizer는 'adam', loss 함수는 'sparse_categorical_entropy' 사용  
* epoch = 15, batch_size = 10\
\
[코드 원문]
```
model = models.Sequential([
        layers.Flatten(input_shape=(input_size_height, input_size_width)),
        layers.Dense(1000, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
#### (2) CNN
* 2쌍의 Convolutional 및 Pooling 층과 Dropout 층, 그리고 Flatten 층과 1개의 은닉층(node 400개, 'ReLU'), 출력층 (node 2개, 'Softmax')으로 구성
* optimizer 및 loss 함수, epoch 및 batch_size는 ANN과 동일\
\
[코드 원문]
```
model = models.Sequential([
        layers.Conv2D(16, kernel_size=(5,5), strides=(1,1), padding='same',
                      activation='relu', input_shape=(input_size_height, input_size_width, depth)),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        layers.Conv2D(32, (2,2), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(400, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
### 4) 성능평가 (1차) (최고 accuracy, 최저 loss 기준)
* ANN: validation accuracy = 0.7656, val_loss = 0.5560
* CNN: validation accuracy = 0.8438, val_loss = 0.3770  
  → validation_accuracy (붉은색)가 낮을 뿐 아니라, validation_loss (초록색)가 발산하며 낮은 성능을 보임  
<img width="754" alt="스크린샷 2021-06-13 오후 3 36 34" src="https://user-images.githubusercontent.com/78459305/121797739-259cf400-cc5d-11eb-8f37-6d3b43fd1c50.png">  

### 5) 전처리 (2차)
* 1차와 동일한 threshold로 이미지 이진화(binarization)  
* resize 1차와 동일 (원본 2160 x 2880 → 360 x 480 으로 축소)  
* dilate와 erode를 반복하여 노이즈 제거 및 ox 이미지 강조      
  (dilate → erode → dilate 순으로 진행, dilate와 erode의 커널 크기는 각각 (45,45), (35,35)로 진행)  
<img width="754" alt="스크린샷 2021-06-13 오후 2 43 17" src="https://user-images.githubusercontent.com/78459305/121796652-b243b400-cc55-11eb-8fe5-60c1103fc038.png">  

* 이미지 내 O, X의 영역만 crop, resize (360 x 360) 후 검은 정사각형 (400 x 400)과 합침  
  → 사각형 모서리의 작은 점 등 노이즈 존재  
<img width="754" alt="스크린샷 2021-06-13 오후 2 55 19" src="https://user-images.githubusercontent.com/78459305/121796877-609c2900-cc57-11eb-85b7-be99af081727.png">

* 이미지 정규화 (0 ~ 255 사이의 pixel값을 0 ~ 1 사이의 값으로 변환)  
* O, X 데이터 통합 및 train/test 데이터셋 분리 (test size = 0.2)

### 6) 성능평가 (2차) (최고 accuracy, 최저 loss 기준)
* 1차와 동일한 모델 사용
* ANN: validation accuracy = 0.9375, val_loss = 10.2692
* CNN: validation accuracy = 0.9219, val_loss = 0.2129  
  → val_accuracy가 높아지고 validation_loss가 수렴하며 1차 대비 성능이 향상되었으나, accuracy가 95% 미만임  
<img width="754" alt="스크린샷 2021-06-13 오후 3 47 25" src="https://user-images.githubusercontent.com/78459305/121797989-a8727e80-cc5e-11eb-8526-27d5d19aaead.png">  
  
### 7) 전처리 (3차)
* threshold를 낮춘 후 이미지 이진화(binarization) (threshold=40)
* resize 1,2차와 동일 (원본 2160 x 2880 → 360 x 480 으로 축소)
* dilate와 erode 2차와 동일  
  → 2차 대비 노이즈 감소
<img width="754" alt="스크린샷 2021-06-13 오후 2 52 10" src="https://user-images.githubusercontent.com/78459305/121796830-eff50c80-cc56-11eb-930e-bf1fda76aaba.png">

* 이미지 crop & resize 또한 2차와 동일  
<img width="754" alt="스크린샷 2021-06-13 오후 2 44 42" src="https://user-images.githubusercontent.com/78459305/121796670-e4551600-cc55-11eb-9335-acd8f254e1a7.png">

* 이미지 정규화 및 데이터 통합, train/test 데이터셋 분리도 2차와 동일  

### 8) 성능평가 (3차) (최고 accuracy, 최저 loss 기준)
* 1,2차와 동일한 모델 사용
* ANN: validation accuracy = 0.9844, val_loss = 0.9687
* CNN: validation accuracy = 0.9844, val_loss = 0.0893  
  → val_accuracy가 98% 이상으로 높아지고 validation_loss도 수렴하며 2차 대비 성능이 향상됨  
<img width="754" alt="스크린샷 2021-06-13 오후 3 57 36" src="https://user-images.githubusercontent.com/78459305/121798221-14a1b200-cc60-11eb-92ca-852fdcde55e8.png">  

----

## 4. 오분류 데이터 확인 및 원인 추정
* 오분류 데이터는 총 1개로 O를 X로 잘못 인식한 경우였으며, 점(노이즈)가 O 반대 방향 꼭짓점에 있는 경우였음  
  → 이미지의 꼭짓점에 데이터가 있으면 X로 분류하는 것으로 추정  
<img width="200" alt="스크린샷 2021-06-13 오후 3 08 55" src="https://user-images.githubusercontent.com/78459305/121797127-46634a80-cc59-11eb-8087-2420aa69b42f.png">   

----

## 5. 추후 보완점
* 노이즈를 더 깨끗하게 제거하기 위한 전처리가 이루어져야 함  
  (e.g. 이진화 threshold 최적화)
----

## 7. 멤버 & 수행업무
#### 1) [임현수](https://github.com/EbraLim/)
* 데이터 수집, 전처리, 모델링 및 성능평가
* 코드 모듈화 및 Github 정리
* 리드미 작성
----

본 프로젝트는 패스트캠퍼스 데이터사이언스 취업스쿨 미니 프로젝트로 진행되었습니다.
