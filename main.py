#from config import main_params, path_params
from embedding import create_embedding
from train import data_classification
from clean import clean_data
from preprocess import make_n_gram, make_tfidf
from eda import make_plot
from test import get_test_predictions
import numpy as np

ngram_values = [1,2,3]
kind_values = ["lemm", "stem", "raw"]
clf_values = ["lr", "svm", "rf", "nb"]

report_list = []

path_params_list = []

def run(ngram_val:int, kind_val:str, clf_val:str, mode:str):

    print(" >>>>>>>>>>>> ngram = " + str(ngram_val) + " | kind = " + kind_val + " | clf = " + clf_val + " | mode = " + mode, end="\n\n")
    """
    mode : can take the following values : "ngram", "tfidf", "raw"
    """
    main_params = {
    "ngram": ngram_val, # options : 1, 2, 3
    "kind": kind_val, # options : lemm or stem or raw
    "clf" : clf_val, # options : lr, rf, svm, nb
    }
    if mode == "ngram":    
        path_params = {
        "RAW_TRAIN_PATH" : "project2_training_data.txt",
        "RAW_TEST_PATH" : "project2_test_data.txt",
        "RAW_LABELS_PATH" : "project2_training_data_labels.txt",

        "TRAIN_LABELS_PATH" : "train_labels.csv",
        "TEST_LABELS_PATH" : "test_labels.csv",
        
        "TEST_PROCESSED_PATH" : "" + main_params["kind"] + "_test.csv",
        "TRAIN_PROCESSED_PATH" : "" + main_params["kind"] + "_train.csv",

        "EMBEDDING_TRAIN_PATH" : "embed_"+main_params["kind"],
        "TRAIN_BEST_PATH" : "ng"+str(main_params["ngram"])+"_"+main_params["kind"]+"_train.csv",
        "TEST_BEST_PATH": "ng1_stem_test.csv",
        }
    elif mode =="tfidf":
        path_params = {
        "RAW_TRAIN_PATH" : "project2_training_data.txt",
        "RAW_TEST_PATH" : "project2_test_data.txt",
        "RAW_LABELS_PATH" : "project2_training_data_labels.txt",

        "TRAIN_LABELS_PATH" : "train_labels.csv",
        "TEST_LABELS_PATH" : "test_labels.csv",
        
        "TEST_PROCESSED_PATH" : "" + main_params["kind"] + "_test.csv",
        "TRAIN_PROCESSED_PATH" : "" + main_params["kind"] + "_train.csv",

        "EMBEDDING_TRAIN_PATH" : "embed_"+main_params["kind"],
        "TRAIN_BEST_PATH" : "tfidf"+"_"+main_params["kind"]+"_train.csv",
        "TEST_BEST_PATH": "tfidf"+"_"+main_params["kind"]+"_test.csv",
        }
    elif mode =="raw":
        path_params = {
        "RAW_TRAIN_PATH" : "project2_training_data.txt",
        "RAW_TEST_PATH" : "project2_test_data.txt",
        "RAW_LABELS_PATH" : "project2_training_data_labels.txt",

        "TRAIN_LABELS_PATH" : "train_labels.csv",
        "TEST_LABELS_PATH" : "test_labels.csv",
        
        "TEST_PROCESSED_PATH" : "" + main_params["kind"] + "_test.csv",
        "TRAIN_PROCESSED_PATH" : "" + main_params["kind"] + "_train.csv",

        "EMBEDDING_TRAIN_PATH" : "embed_"+main_params["kind"],
        "TRAIN_BEST_PATH" : "tfidf_raw_train.csv",
        "TEST_BEST_PATH": "tfidf_raw_test.csv",
        }
    elif mode == "embedding":
        path_params = {
        "RAW_TRAIN_PATH" : "project2_training_data.txt",
        "RAW_TEST_PATH" : "project2_test_data.txt",
        "RAW_LABELS_PATH" : "project2_training_data_labels.txt",

        "TRAIN_LABELS_PATH" : "train_labels.csv",
        "TEST_LABELS_PATH" : "test_labels.csv",
        
        "TEST_PROCESSED_PATH" : "" + main_params["kind"] + "_test.csv",
        "TRAIN_PROCESSED_PATH" : "" + main_params["kind"] + "_train.csv",

        "EMBEDDING_TRAIN_PATH" : "embed_"+main_params["kind"],
        "TRAIN_BEST_PATH" : "ng2_lemm_train.csv", # NOT REQUIRED
        "TEST_BEST_PATH": "ng1_stem_test.csv",
        }

    # cleans train and test texts
    clean_data(path_params, main_params["kind"])
    create_embedding(path_params, main_params["kind"])

    # plotting data from train
    make_plot(path_params, path_params["RAW_TRAIN_PATH"])

    # make n-gram for train and test
    make_n_gram(main_params["ngram"], path_params["TRAIN_PROCESSED_PATH"])
    make_n_gram(main_params["ngram"], path_params["TEST_PROCESSED_PATH"])
    
    # make tfidf for train and test, only required for TFIDF
    make_tfidf(path_params, data_path=path_params["TRAIN_PROCESSED_PATH"], test_data_path=path_params["TEST_PROCESSED_PATH"])

     # run training, cv, and validation to get best model and params
    report_list.append([[ngram_val, kind_val, clf_val, mode], data_classification(path=path_params["TRAIN_BEST_PATH"], no_of_selected_features = 99, clf_opt = main_params["clf"]).classification()])
    path_params_list.append(path_params)

    if mode == "embedding":
        data_classification(path=path_params["EMBEDDING_TRAIN_PATH"], no_of_selected_features = 99, clf_opt = main_params["clf"], embedding = True).classification()
        pass
   # get_test_predictions(test_data_path=path_params["TEST_BEST_PATH"], train_data_path=path_params["TRAIN_BEST_PATH"])

# Main code
if __name__ == "__main__":
    # Running N Gram 
    for ngram_value in ngram_values:
        for kind_value in kind_values[:2]:
            for clf_value in clf_values:
                run(ngram_value, kind_value, clf_value, mode="ngram")
    
    # Running TFIDF
    for kind_value in kind_values[:2]:
            for clf_value in clf_values:
                run(1, kind_value, clf_value, mode="tfidf")

    # Running Raw
    for clf_value in clf_values:
        run(1, "raw", clf_value, mode="raw")
    

    # Running the prediction
    accuracies_list = [] # F1 Score Accuracy List
    f1_list = [] # Macro Avg F1 Score List
    mavg_prec_list = [] # Macro Avg Precision List

    for report in report_list:
        print(report[1])
        accuracies_list.append(float(report[1]["accuracy"]))
        f1_list.append(float(report[1]["macro avg"]["precision"]))
        mavg_prec_list.append(float(report[1]["macro avg"]["recall"]))

    top_accuracies_indices = np.where(np.array(accuracies_list)==max(accuracies_list))[0] 
    top_f1_scores = []
    for index in top_accuracies_indices:
        top_f1_scores.append([index, f1_list[index]])
    
    ## Get the top-most f1 scores
    _ = []
    for entry in top_f1_scores:
        _.append(entry[1])
    top_f1_score_indices = np.where(np.array(_)==max(_))[0] 

    top_mavg_prec_list = []
    for index in top_f1_score_indices:
        top_mavg_prec_list.append([top_f1_scores[index][0], mavg_prec_list[top_f1_scores[index][0]]])

    _ = []
    for entry in top_mavg_prec_list:
        _.append(entry[1])

    top_mavg_prec_index = mavg_prec_list.index(max(_))

    # Running Embedding
    for kind_value in kind_values[:2]:
            for clf_value in clf_values[:3]:
                run(1, kind_value, clf_value, mode="embedding")    

    print("\n\n############  The top performer was : " , report_list[top_mavg_prec_index][0], end="\n\n")
    get_test_predictions(path_params_list[top_mavg_prec_index],test_data_path=path_params_list[top_mavg_prec_index]["TRAIN_BEST_PATH"], train_data_path=path_params_list[top_mavg_prec_index]["TRAIN_BEST_PATH"])
