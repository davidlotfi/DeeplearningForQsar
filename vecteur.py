import pandas as pd
from gensim.models import Word2Vec
dataset=pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])
SmilesCanonical=dataset.iloc[:,7].values
Observed=dataset.iloc[:,1].values
#print(SmilesCanonical[218])
print('model de transformation')
modelV = Word2Vec.load("m.model")
print(f'nomber de molecule : {modelV.corpus_count}' )
#print(modelV)
SmilesCanonical2 = modelV[modelV.wv.vocab]
#print(SmilesCanonical2[:20])
print('la boucle -----------------------')


for str in SmilesCanonical[:1]:
    print(f'la permier molécule : {str}')


print('parccourir les chaine de caracter dans la permier molécule :')

for i in range(len(str)):  #parc
    print(str[i])







print(str[0])
print('predict------------------------------------------')

print(modelV.predict_output_word(str[0]))

#vec=modelV.predict_output_word(str[0])
