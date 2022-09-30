import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
#from config import path_params

def make_plot(path_params, path):

    #pull the data
    # Read the training data
    msg = pd.read_csv(path_params["TRAIN_PROCESSED_PATH"])
    #msg.columns = ['msg']

    # label of training dataset
    data = pd.read_csv(path_params["RAW_LABELS_PATH"], sep='\t', header=None)
    data.columns = ['label']
    #WordCloud

    wordcloud = WordCloud(width = 1000, height = 500).generate(" ".join(msg['texts']))
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('worldcloud.png', dpi=300)
    plt.show()

    # Bar plot
    fig = plt.figure(figsize=(8,6))
    data.groupby('label')['label'].count().plot.bar(ylim=0)
    plt.savefig('Barplot.png', dpi=300)
    plt.show()

    # Confusion Matrix 
def make_confusion_matrix(map):

    fig = plt.figure(figsize=(8,6))
    sns.heatmap(map)
    plt.savefig('Confusion_Matrix_Train.png', dpi=300)
    plt.show()
