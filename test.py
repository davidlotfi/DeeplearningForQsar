import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from gensim.models import Word2Vec
from nltk.corpus import *

dataset=pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])
SmilesCanonical=dataset.iloc[:,7].values
Observed=dataset.iloc[:,1].values
print('model de transformation')
modelV = Word2Vec.load("m.model")
print(modelV)
list = []
for str in SmilesCanonical:
 vec = []
 for s in str: #erreur
   print(s)
   vec =  sum( modelV.predict_output_word(str[s]),vec)
   list.append(vec)
 print(list)
X = list


""""

X = X.astype('float64')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X = X.reshape(X.shape[0], 100, 1, 1)
model= Sequential()
model.add(Conv2D (kernel_size = (2,1), filters = 1, input_shape=(100,1,1), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,1), strides=(2,1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
print(model.summary())
model.compile(loss='mse', optimizer='adam',metrics=['mse'])#metrics la zam charh
print('compile model')
print('fit model')
model.fit(X ,Observed,batch_size=11,epochs=100)

"""