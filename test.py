import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
#from config import path_params

# BEST PARAMS = {'clf__alpha': 1} in  nb

def get_test_predictions(path_params, train_data_path, test_data_path):

    # load model
    # best_clf = MultinomialNB(fit_prior=True, class_prior=None, alpha=1)
    best_clf = LogisticRegression(random_state = 0)
    test_data = pd.read_csv(test_data_path)
    train_data = pd.read_csv(train_data_path)
    common_columns = []
    for fcols in train_data.columns.tolist():
        if fcols in test_data.columns.tolist():
            common_columns.append(fcols)
    test_data = test_data[common_columns].to_numpy()
    train_data = train_data[common_columns].to_numpy()

    labels = pd.read_csv(path_params["TRAIN_LABELS_PATH"]).encoded_label.to_numpy()
    
    best_clf.fit(train_data, labels)

    # save predictions
    pd.DataFrame(best_clf.predict(test_data)).to_csv(path_params["TEST_LABELS_PATH"], index = None)
