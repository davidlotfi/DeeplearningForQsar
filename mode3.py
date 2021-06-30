import pandas as pd
from keras.models import Sequential
from keras.layers import *
from gensim.models import Word2Vec
from nltk.corpus import *

#load data
dataset=pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])
SmilesCanonical=dataset.iloc[:,7:8].values
Observed=dataset.iloc[:,1].values


##Encoding categorical data    : normalement le princip de NLP  text >>>> number
## Encoding the Independent Variable : Independent les variable string dans cette exemple on a SmilesCanonical

print('model de transformation')

modelV = Word2Vec.load("word2vec.model2")  #model2 est bien

SmilesCanonical2 = modelV[modelV.wv.vocab]
print(Observed[:20])
print(SmilesCanonical2[:20])
X=SmilesCanonical2.ravel()
print(X)
t=modelV.most_similar(positive='Cc1ccccc1N', topn=1)  #pour trouve les molecul similaire
print(t)
print('----------------------------------------------')

print(X.shape)

input_shape = (1,)

#premier definie architecture de modele et le nomber de couche

print('model ----------------------------------')
model= Sequential()
model.add(Dense(1, input_shape=(1,)))

print(model.summary())
#compile le modele


model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
print('compile model')

#fit model in training data
print('fit model')
#model.fit(X ,Observed,batch_size=1,epochs=10)

#model.fit(data,epochs=10, batch_size=32)
