[Deep Learning project, toy project]\
OX classification with hand-written ox images
==
## 1. 소개
### 1) 주제
* 손글씨로 생성한 O, X 이미지를 분류하는 딥러닝 모델 만들기
### 2) 목표
* 적절한 데이터 전처리를 통한 학습데이터 품질 및 모델 성능 향상 (accuracy 95% 이상)
### 3) 결과
* 최종 test accuracy: 95 ~ 97%
----

## 2. 진행 프로세스
<img width="233" alt="스크린샷 2021-05-21 오후 12 07 06" src="https://user-images.githubusercontent.com/78459305/119076168-11ceeb00-ba2d-11eb-9e8b-70f441f8cfca.png">
* 총 2번의 iteration 수행
----

## 3. 프로세스별 상세설명
### 1) 데이터 수집
* A4 용지를 64등분한 크기로 O,X 각각 320개씩 제작 후 직접 촬영
* 데이터 복잡성 향상을 위해 선 굵기가 다른 5개의 펜 사용하였으며, 필적 학습 방지를 위해 양손을 번갈아가며 제작

### 2) 전처리 (1차)
* 이미지 이진화(binarization) (threshold=50)
* resize (원본 2160 x 2880 → 360 x 480 으로 축소)
* 딥러닝 학습을 위한 scaling (0~255 사이의 값을 0~1 사이의 값으로 변환)
* O, X 데이터 통합 및 train/test 데이터셋 분리 (test size = 0.3)

### 3) 모델링
* ANN과 CNN, 총 두가지 모델 사용
#### (1) ANN
- 알고리즘 도식(이미지 삽입 예정)
- optimizer는 'adam', loss 함수는 'sparse_categorical_entropy' 사용함
- epoch = 10, batch_size = 10
#### (2) CNN
- 알고리즘 도식(이미지 삽입 예정)
- optimizer 및 loss 함수는 ANN과 동일
- epoch = 5

### 4) 성능평가 (1차)
* ANN: validation accuracy = 0.7083, val_loss = 0.7826, 이후 발산
* CNN: validation accuracy = 0.7917, val_loss = 0.8674, 이후 발산

### 5) 전처리 (2차)
* threshold 낮추어 이미지 이진화(binarization) (threshold=20)
* dilate와 erode를 반복하여 노이즈 제거 및 ox 이미지 강조    
  (dilate → erode → dilate 순으로 진행, dilate와 erode의 커널 크기는 각각 (45,45), (35,35)로 진행)
* 이미지 내 O, X의 영역만 crop, resize (360 x 360) 후 검은 정사각형 (400 x 400)과 합침
* 딥러닝 학습을 위한 scaling (0~255 사이의 값을 0~1 사이의 값으로 변환)
* O, X 데이터 통합 및 train/test 데이터셋 분리 (test size = 0.3)

### 6) 성능평가 (2차)
* 1차와 동일한 모델 사용
* ANN: validation accuracy = 0.9688, val_loss = 7.0433
* CNN: validation accuracy = 0.9531, val_loss = 0.2887  
  → 1차 대비 성능이 크게 향상됨
----

## 4. 오분류 데이터 확인 및 원인 추정
* Step 별로 결과물 이미지 or 움짤 or 유튜브링크 첨부 예정
----

## 6. 추후 보완점
* 작성예정
----

## 7. 멤버 & 수행업무
#### 1) [임현수](https://github.com/EbraLim/)
* 작성예정
#### 2) [정민주](https://github.com/meiren13/)
* 작성예정
----

본 프로젝트는 패스트캠퍼스 데이터사이언스 취업스쿨 16th 딥러닝 프로젝트로 진행되었습니다.
