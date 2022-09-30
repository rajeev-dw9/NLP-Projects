from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
#from config import path_params

def make_n_gram(n_gram=2, data_path=None):

    data = pd.read_csv(data_path)
    vectorizer_ngram = CountVectorizer(min_df=5, stop_words='english', ngram_range=(n_gram, n_gram))
    print(data.columns)
    ngram_data = vectorizer_ngram.fit_transform(data.texts.values.astype('U'))
    features = vectorizer_ngram.get_feature_names_out()

    # bag of words
    bagofwords_ngram = pd.DataFrame.sparse.from_spmatrix(ngram_data, columns=features)
    # save the file
    bagofwords_ngram.to_csv("ng" + str(n_gram) + '_' + data_path.split('/')[-1].split('.')[0] +".csv", index=False)

def make_tfidf(path_params, data_path = None, test_data_path = None):
    data = pd.read_csv(data_path)
    vector_tfidf_maker = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
    tfidf_data = vector_tfidf_maker.fit_transform(data.texts.values.astype('U')).toarray()
    labels = pd.read_csv(path_params["TRAIN_LABELS_PATH"]).drop("Unnamed: 0", axis=1)
    skb = SelectKBest(chi2, k=100)
    tfidf_df = pd.DataFrame(skb.fit_transform(tfidf_data, labels.values))
    tfidf_df.to_csv("tfidf_" + data_path.split('/')[-1].split('.')[0] + ".csv", index=False)
    
    test_data = pd.read_csv(test_data_path)
    tfidf_test_data = vector_tfidf_maker.transform(test_data.texts.values.astype('U')).toarray()
    tfidf_test_df = pd.DataFrame(skb.transform(tfidf_test_data))
    # save the file
    tfidf_test_df.to_csv("tfidf_" + test_data_path.split('/')[-1].split('.')[0] + ".csv", index=False)