# basic settings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import time
import tensorflow as tf
from tensorflow.keras import layers, models
# %matplotlib inline
plt.rc('font', family='AppleGothic')

# raw data 가져오기
def import_data(type="both"):
    
    """
    one output for one type, either "o" or "x", 
    and two outputs for type "both"
    """

    o_list = []
    o_file_list = np.arange(3544, 3705)
    x_list = []
    x_file_list = np.arange(3705, 3865)
    start_time = time.time()
    
    if type=="o":
        for num in o_file_list:
            o_image = cv2.imread(f'oxdata/o/IMG_{num}.JPG')
            try:
                # Nonetype 걸러내기 위함
                _ = len(o_image)
                # 가로가 세로보다 길 경우, 시계방향으로 90도 회전
                if o_image.shape[0] < o_image.shape[1]:
                    o_image = cv2.rotate(o_image, cv2.ROTATE_90_CLOCKWISE)
                # 이미지가 너무 커 사이즈 축소 ((2160, 2880) -> (360, 480))
                o_image = cv2.resize(o_image, (360, 480), interpolation=cv2.cv2.INTER_NEAREST)
                o_list.append(o_image)
            except:
                pass
        print("Loading Time: ", time.time() - start_time)
        print("Loading completed!")
        return o_list
        
    elif type=="x":
        for num in x_file_list:
            x_image = cv2.imread(f'oxdata/x/IMG_{num}.JPG')
            try:
                # Nonetype 걸러내기 위함                
                _ = len(x_image)
                # 가로가 세로보다 길 경우, 시계방향으로 90도 회전
                if x_image.shape[0] < x_image.shape[1]:
                    x_image = cv2.rotate(x_image, cv2.ROTATE_90_CLOCKWISE)
                # 이미지가 너무 커 사이즈 축소 ((2160, 2880) -> (360, 480))
                x_image = cv2.resize(x_image, (360, 480), interpolation=cv2.cv2.INTER_NEAREST)                
                x_list.append(x_image)    
            except:    
                pass
        print("Loading Time: ", time.time() - start_time)
        print("Loading completed!")
        return x_list
    
    else:
        for num in o_file_list:
            o_image = cv2.imread(f'oxdata/o/IMG_{num}.JPG')
            try:
                # Nonetype 걸러내기 위함
                _ = len(o_image)
                # 가로가 세로보다 길 경우, 시계방향으로 90도 회전
                if o_image.shape[0] < o_image.shape[1]:
                    o_image = cv2.rotate(o_image, cv2.ROTATE_90_CLOCKWISE)
                # 이미지가 너무 커 사이즈 축소 ((2160, 2880) -> (360, 480))
                o_image = cv2.resize(o_image, (360, 480), interpolation=cv2.cv2.INTER_NEAREST)
                o_list.append(o_image)
            except:
                pass
            
        for num in x_file_list:
            x_image = cv2.imread(f'oxdata/x/IMG_{num}.JPG')
            try:
                # Nonetype 걸러내기 위함                
                _ = len(x_image)
                # 가로가 세로보다 길 경우, 시계방향으로 90도 회전
                if x_image.shape[0] < x_image.shape[1]:
                    x_image = cv2.rotate(x_image, cv2.ROTATE_90_CLOCKWISE)
                # 이미지가 너무 커 사이즈 축소 ((2160, 2880) -> (360, 480))
                x_image = cv2.resize(x_image, (360, 480), interpolation=cv2.cv2.INTER_NEAREST)                
                x_list.append(x_image)    
            except:    
                pass
            
        print("Loading Time: ", time.time() - start_time)
        print("Loading completed!")
        return o_list, x_list
    
    
# 이진화하기
def binarize(data, low=50):
    
    start_time = time.time()    
    
    for num in range(len(data)):
        img_gray = cv2.cvtColor(data[num], cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img_gray, low , 255, cv2.THRESH_BINARY_INV)
        data[num] = img_binary
        
    print("Time for binarization:", time.time() - start_time)
    print("Binarization completed!")
    return data


# 전처리: 수축 및 팽창하기
def dilate_and_erode(data):
    
    # 각 커널 세팅
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    
    for num in range(len(data)):
        
        # dilate 2번, erode 한 번 해서 노이즈는 줄이고 ROI는 확보하기
        dilated = cv2.dilate(data[num], kernel_dilate)
        eroded = cv2.erode(dilated, kernel_erode)
        dilated_2 = cv2.dilate(eroded, kernel_dilate)
        data[num] = dilated_2
    
    print("dilation and erosion completed!")
    return data


# ROI 추출 후 정방 사각형으로 사이즈 변환
def roi_extract(data, roi_size=360, entire_size=400):
    
    for num in range(len(data)):
        
        # roi 추출
        hor = np.sum(data[num], axis=1)
        ver = np.sum(data[num], axis=0)
        roi = data[num][np.min(np.where(hor>0)):np.max(np.where(hor>0)), 
                        np.min(np.where(ver>0)): np.max(np.where(ver>0))]
        roi = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_NEAREST)
        
        # 검정 정사각형에 얹기
        black = np.full((entire_size, entire_size), 0, np.uint8)
        black_crop = black[20:(entire_size-20), 20:(entire_size-20)]
        crop_added = cv2.add(black_crop, roi)
        black[20:(entire_size-20), 20:(entire_size-20)] = crop_added
        
        # 원데이터 치환       
        data[num] = black
        
    print("ROI extraction completed!")
    return data


# 데이터 통합 및 훈련/테스트 데이터 분리
def concat_and_split(data1, data2, test_size=0.2):
    total = np.concatenate((data1, data2), axis=0)
    y_label = np.concatenate(([1]*len(data1), [0]*len(data2)), axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(total, y_label, test_size = test_size, random_state=13)
    
    # normalization
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    print("Concatenation and split completed!")
    return X_train, X_test, y_train, y_test


# ANN 모델 만들기
def create_ANN_model(input_size_width=400, input_size_height=400):
    start_time = time.time()
    
    model = models.Sequential([
        layers.Flatten(input_shape=(input_size_height, input_size_width)),
        layers.Dense(1000, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Fit Time: ", time.time() - start_time)
    print("ANN model is created!")
    return model


# CNN 모델 만들기
def create_CNN_model(input_size_width=400, input_size_height=400, depth=1):
    start_time = time.time()
    
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
    
    print("Fit Time: ", time.time() - start_time)
    print("CNN model is created!")
    return model


# ANN 결과 도출
def get_result_ann(X_train, X_test, y_train, y_test, model, batch_size=0, epochs=15, verbose=1, graph_show=True):
    
    start_time = time.time()
    if batch_size:
        hist = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test),
                         epochs=epochs, verbose=verbose)
    else:
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=verbose)
    
    print("Fit Time: ", time.time() - start_time)
    print("result completed!")
    
    if graph_show:
        plot_target = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        plt.figure(figsize=(12,8))

        for each in plot_target:
            plt.plot(hist.history[each], label=each)

        plt.legend()
        plt.grid(True)
        plt.show()
    
    return hist


# CNN 결과 도출
def get_result_cnn(X_train, X_test, y_train, y_test, model, batch_size=0, epochs=15, verbose=1, depth=1, graph_show=True):
    
    start_time = time.time()
    
    X_tr_reshape = X_train.shape + (depth,)
    X_te_reshape = X_test.shape + (depth,)
    X_train, X_test  = X_train.reshape(X_tr_reshape), X_test.reshape(X_te_reshape)
    
    print(X_train.shape, X_test.shape)
    if batch_size:
        hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test))
    else:
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=verbose)
    
    print("Fit Time: ", time.time() - start_time)
    print("result completed!")
    
    if graph_show:
        plot_target = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        plt.figure(figsize=(12,8))

        for each in plot_target:
            plt.plot(hist.history[each], label=each)

        plt.legend()
        plt.grid(True)
        plt.show()
    
    return hist


# 전처리 (이진화, 팽창/수축, ROI 추출 등 한 번에)
def preprocessing(data):
    
    start_time = time.time()
    
    data = binarize(data)
    data = dilate_and_erode(data)
    data = roi_extract(data)
    
    print("Time for preprocessing:", time.time() - start_time)
    print("Preprocessing completed!")
    return data