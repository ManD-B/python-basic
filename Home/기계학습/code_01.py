import tensorflow as tf
import numpy as np

# seed 값 설정
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils



# 데이터 입력
data = np.loadtxt('HW01_dataset.csv', delimiter=",")

X = data[:,1:10] # 속성
Y = data[:,10] # 클래스
#Y = Y - 1
Y_encoded = np_utils.to_categorical(Y)

model = Sequential()	# 모델 선언
model.add(Dense(16, input_dim=9, activation='relu'))	
model.add(Dense(64, activation='relu'))	
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))	
model.add(Dense(7, activation='softmax'))	

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y_encoded, epochs=300, batch_size=25)

print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))