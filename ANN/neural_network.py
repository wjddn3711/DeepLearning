# Artificial Neural Network

# Importing the libraries
# 데이터 전처리 템플릿 - numpy, pandas import
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)      # 현재 텐서플로 버젼 체크

# Part 1 - Data Preprocessing
# 데이터 전처리 - 전체 구현의 첫번째 파트
# Importing the dataset - 프로젝트 내에 파일 가져오기
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values # 마지막 열을 제외한 모든 열을 사용해서 특성 매트릭스 X 를 만듦
y = dataset.iloc[:, -1].values # 마지막 열이 결과값이 됨
# Encoding categorical data
# Label Encoding the "Gender" column
# 성별의 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) # 모든 행의 인덱스 2 인 성별을 인코딩
# One Hot Encoding the "Geography" column
# 국가의 라벨 인코딩 - 개별적이며 서로 관계가 없기 때문에 원핫인코딩 실시
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
# 데이터 세트를 훈련 세트로 분리해서 테스트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN
# Initializing the ANN
# ANN을 일련의 층으로 초기화
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# 입력층과 함께 결정한 수만큼의 뉴런으로 구성된 첫번째 레이어 추가

# 뉴런의 개수를 어떻게 정할까? 경험? 실험? 실험을 바탕으로 해야함
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # 정규화 활성화 함수로 은닉 뉴런을 6개로 은닉층을 만든다

# Adding the second hidden layer
# 얇지 않은 딥러닝 모델을 구축하기 위해 두번째 레이어 추가
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
# 예측 결과가 나올 출력층을 추가
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN
# Compiling the ANN
# ANN 컴파일링
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
# ANN 훈련시키기
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Part 4 - Making the predictions and evaluating the model
# 예측 및 모델 평가
# Predicting the result of a single observation
# 예측 메소드의 입력값은 무조건 2D 배열이어야 함
# 원핫 인코딩을 적용한 이후의 프랑스는 1,0,0 의 더미 값을 가진다 / 성별의 경우 남성 1, 여성 0 임을 기억
# 예측 메소드는 훈련할 때와 같은 스케일링이 적용된 관측치를 사용해야 한다는 점을 주의. 즉, 정규화를 거쳐야함
# fit_transform 은 스케일러를 훈련 세트에 맞추기위해 값과 훈련세트의 평균 및 표준편차를 구하는데 사용되기 때문에 이경우 사용되지 않음
# 새 관측치, 실제 생산 과정에 모델을 배포하는 경우 변환 메서드만 적용할 수 있다는 점을 유의
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5) # 만약 50% 이상의 이탈율을 보인다면 결과를 1로 봄

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
