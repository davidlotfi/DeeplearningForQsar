import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy
def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
#dataset = pd.read_csv('trainingi50.txt',delimiter=';')
#datasett = pd.read_csv('testi50.txt',delimiter=';')
dataset = pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])
datasett = pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])

X = dataset.iloc[:, 1:129].values
y = dataset.iloc[:, 0].values
X = X.astype('float32')
X_test = datasett.iloc[:, 1:129].values
y_test = datasett.iloc[:, 0].values
X_test = X_test.astype('float32')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X = X.reshape(X.shape[0], 16, 8, 1)
X_test = sc.transform(X_test)
X_test = X_test.reshape(X_test.shape[0],16,8,1)
input_shape = (16, 8, 1)
import keras
from keras.models import Sequential
from keras.layers import *
model= Sequential()
model.add(Conv2D(32, kernel_size=5, padding="same",input_shape= input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, kernel_size=2, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=[coeff_determination,'mse','mae'])
model.fit(X,y,batch_size=len(X),nb_epoch=1000)
model_json = model.to_json()
with open("3DQsarRegression.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("3DQsarRegression.h5")
print("Saved model to disk")
print("validated performances")
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)
print(model.predict(X))
