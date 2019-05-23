
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import *
import pandas as pd


dataset=pd.read_csv('ic.csv', sep=';', engine='python',
    na_values=['NA','?'])
SmilesCanonical=dataset.iloc[:,7].values

model_1 = Word2Vec(size=300, min_count=1)

model_1.build_vocab(SmilesCanonical)

total_examples = model_1.corpus_count
print(total_examples)
SmilesCanonical2 = model_1[model_1.wv.vocab]

print(SmilesCanonical2 )
