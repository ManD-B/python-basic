# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옵니다.
import numpy

# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = numpy.loadtxt("ThoraricSurgery.csv", delimiter=",")

X = Data_set[:,0:17] # 속성
Y = Data_set[:,17] # 클래스

model = Sequential()	# 모델 선언
model.add(Dense(30, input_dim=17, activation='relu'))	
# 입력층과 은닉층 설계 : 데이터에서 17개의 값을 받아 은닉층의 30개 노드로 보낸다는 뜻
# 렐루(relu) 함수를 활성화 함수로 사용
model.add(Dense(1, activation='sigmoid'))	
# 출력층 설계 : 출력층의 노드 수는 1개, 시그모이드 함수를 활성화 함수로 사용

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X, Y, epochs=30, batch_size=10)
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))



