import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import argparse
#from config import path_params

# import the library
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')


def clean_data(path_params, kind="lemm"):

  # Read the training data
  msg = pd.read_csv(path_params["RAW_TRAIN_PATH"], sep="\t", header=None)
  test_msg = pd.read_csv(path_params["RAW_TEST_PATH"], sep="\t", header=None)
 
  # label of training dataset
  data = pd.read_csv(path_params["RAW_LABELS_PATH"], sep='\t', header=None)
  data.columns = ['label']

  # label
  data['encoded_label'] = data['label'].factorize()[0]
  data["encoded_label"].to_csv(path_params["TRAIN_LABELS_PATH"])

  data["texts"] = msg

  # removing the stop words and,uppercase,lemmatizing

  if kind == "stem":

    porter = PorterStemmer()
    corpus_1 = []
    for i in range(0, len(data)):
      new_data = re.sub('[^a-zA-Z]', ' ', data["texts"][i])
      new_data= new_data.lower()
      new_data = new_data.split()
      new_data = [porter.stem(word) for word in new_data if not word in stop_words]
      new_data = ' '.join(new_data)
      corpus_1.append(new_data)

    corpus_1_test = []
    for i in range(0, test_msg.shape[0]):
      
      new_test_data = re.sub('[^a-zA-Z]', ' ', test_msg[0][i])
      new_test_data= new_test_data.lower()
      new_test_data = new_test_data.split()
      new_test_data = [porter.stem(word) for word in new_test_data if not word in stop_words]
      new_test_data = ' '.join(new_test_data)
      corpus_1_test.append(new_test_data)

    # after porter stemming dataset
    stemmed = pd.DataFrame(corpus_1).rename(columns = {0:"texts"})
    stemmed["label"] = data["label"]
    stemmed.to_csv(path_params["TRAIN_PROCESSED_PATH"], index = False)

    # after porter stemming dataset
    stemmed = pd.DataFrame(corpus_1_test).rename(columns = {0:"texts"})
    stemmed.to_csv(path_params["TEST_PROCESSED_PATH"], index = False)

  elif kind == "lemm":
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    lemmatizer = WordNetLemmatizer()

    corpus_2 = []
    for i in range(0, len(data)):
      message_2 = re.sub('[^a-zA-Z]', ' ', data["texts"][i])
      message_2 = message_2.lower()
      message_2 = message_2.split()
      message_2 = [lemmatizer.lemmatize(word) for word in message_2 if not word in stop_words]
      message_2 = ' '.join(message_2)
      corpus_2.append(message_2)
    lemmatized = pd.DataFrame(corpus_2).rename(columns = {0:"texts"})
    lemmatized["label"] = data["label"]
    lemmatized.to_csv(path_params["TRAIN_PROCESSED_PATH"], index = False)

    corpus_1_test = []
    for i in range(0, test_msg.shape[0]):
      
      new_test_data = re.sub('[^a-zA-Z]', ' ', test_msg[0][i])
      new_test_data= new_test_data.lower()
      new_test_data = new_test_data.split()
      new_test_data = [lemmatizer.lemmatize(word) for word in new_test_data if not word in stop_words]
      new_test_data = ' '.join(new_test_data)
      corpus_1_test.append(new_test_data)

    # after lemmatizing dataset
    lemmatized = pd.DataFrame(corpus_1_test).rename(columns = {0:"texts"})
    lemmatized.to_csv(path_params["TEST_PROCESSED_PATH"], index = False)

  else:

    corpus_1 = []
    for i in range(0, len(data)):
      new_data = re.sub('[^a-zA-Z]', ' ', data["texts"][i])
      new_data= new_data.lower()
      new_data = new_data.split()
      new_data = [word for word in new_data if not word in stop_words]
      new_data = ' '.join(new_data)
      corpus_1.append(new_data)

    corpus_1_test = []
    for i in range(0, test_msg.shape[0]):
      
      new_test_data = re.sub('[^a-zA-Z]', ' ', test_msg[0][i])
      new_test_data= new_test_data.lower()
      new_test_data = new_test_data.split()
      new_test_data = [word for word in new_test_data if not word in stop_words]
      new_test_data = ' '.join(new_test_data)
      corpus_1_test.append(new_test_data)

    raw = pd.DataFrame(corpus_1).rename(columns = {0:"texts"})
    raw["label"] = data["label"]
    raw.to_csv(path_params["TRAIN_PROCESSED_PATH"], index = False)

    raw = pd.DataFrame(corpus_1_test).rename(columns = {0:"texts"})
    raw.to_csv(path_params["TEST_PROCESSED_PATH"], index = False)

  print("Data cleaning done!!")
