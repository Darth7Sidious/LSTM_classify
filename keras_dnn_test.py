from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import numpy as np

model = Sequential()
model.add(Dense(units=64,input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

#生成1以内随机数二维数组，1000行100列
data = np.random.random((1000, 100))
#生成10以内的整数二维数组，1000行1列
labels = np.random.randint(10, size=(1000, 1))

#将整型标签转为onehot
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=2, batch_size=32)
score = model.evaluate(data, one_hot_labels, batch_size=32)
print(score)

data_test = np.random.random((10, 100))
predict_test = model.predict(data_test)
predict = np.argmax(predict_test,axis=1)  #axis = 1是取行的最大值的索引，0是列的最大值的索引

print(predict_test)
print(np.argmax(predict_test,axis=1))

