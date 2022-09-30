import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gensim
from gensim.models.word2vec import Word2Vec
import pandas as pd
import csv, os, re, sys, codecs
import numpy as np
import nltk
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#from config import path_params, main_params
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize

labelencoder = LabelEncoder()
n_dim=256

def buildWordVector(model, tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model.wv[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

def create_embedding(path_params, kind):

    if kind == "lemm":

        ## read the data
        lemm_train_data = pd.read_csv(path_params["TRAIN_PROCESSED_PATH"])
        #lemm_test_data = pd.read_csv('C:/Users/pushpendra/Desktop/Sentiment-Classification-main/dataset/processed/lemm_test.csv')
        
        lemm_train_data['label'] = labelencoder.fit_transform(lemm_train_data['label'])
        lemm_train_data['tokenized_sents'] = lemm_train_data.apply(lambda row: nltk.word_tokenize(row['texts']), axis=1)
        vec_lemm = Word2Vec(sentences=lemm_train_data.tokenized_sents, vector_size=n_dim, min_count=10)
        words_vec_lemm = list(vec_lemm.wv.index_to_key)
        vec_lemm = np.concatenate([buildWordVector(vec_lemm, z, n_dim) for z in map(lambda x: x, lemm_train_data.tokenized_sents)])
        vec_lemm = scale(vec_lemm)
        pd.DataFrame(vec_lemm).to_csv(path_params["EMBEDDING_TRAIN_PATH"][:-4] + "lemm", index = None)

    elif kind == "stem":
        
        ## read the data
        stem_train_data= pd.read_csv(path_params["TRAIN_PROCESSED_PATH"])
        # factorization of the labels   
        stem_train_data['label'] = labelencoder.fit_transform(stem_train_data['label'])
        #stem_test_data = pd.read_csv('C:/Users/pushpendra/Desktop/Sentiment-Classification-main/dataset/processed/stem_test.csv')
        stem_train_data['tokenized_sents'] = stem_train_data.apply(lambda row: nltk.word_tokenize(row['texts']), axis=1)
        vec_stem = Word2Vec(sentences=stem_train_data.tokenized_sents, vector_size=n_dim, min_count=10)
        words_vec_stem = list(vec_stem.wv.index_to_key)
        vec_stem = np.concatenate([buildWordVector(vec_stem, z, n_dim) for z in map(lambda x: x, stem_train_data.tokenized_sents)])
        vec_stem = scale(vec_stem)
        pd.DataFrame(vec_stem).to_csv(path_params["EMBEDDING_TRAIN_PATH"][:-4] + "stem", index = None)

    else:
        print("Raw data cannot be used for embedding due to explosive memory.")
##word tokenization for stem data
# stem_train_data['tokenized_sents'][0]
# stem_test_data['tokenized_sents'] = stem_test_data.apply(lambda row: nltk.word_tokenize(row['texts']), axis=1)

##word tokenization for lemm data
# lemm_test_data['tokenized_sents'] = lemm_test_data.apply(lambda row: nltk.word_tokenize(row['texts']), axis=1)
#print(lemm_train_data['tokenized_sents'][0])
