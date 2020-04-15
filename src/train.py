import pandas as pd
import sys
import logging
import numpy as np
from src.utils.utils import *
from src.utils import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import random
from sklearn.ensemble import AdaBoostClassifier
model_output_dir = "models/"
logging.basicConfig(level=logging.INFO)

def tfidf_features( train, test, type):
    global model_output_dir
    #tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=10, max_features=1000)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=10, max_features=1000)
    train = tfidf_vectorizer.fit_transform(train)
    test = tfidf_vectorizer.transform(test)
    pickle.dump(tfidf_vectorizer, open(model_output_dir+type + "_tfidf_vectorizer.pickle", "wb"))
    return train, test


def numeric_features(train, test):
    # print(train.columns)
    train = train[['title_len', 'question_len', 'code_block_len', 'nums_code_block', 'url_count', 'so_url_count']]
    test = test[['title_len', 'question_len', 'code_block_len', 'nums_code_block', 'url_count', 'so_url_count']]
    return train, test


def transform_features(train, test):
    logger = logging.getLogger("train.transform_features")

    logger.info("generating tf-idf features")
    # get tf-idf features from question body
    train_tfidf_body, test_tfidf_body = tfidf_features(train['clean_body'], test['clean_body'], "question")

    # get tf-idf features from question title
    train_tfidf_title, test_tfidf_title = tfidf_features(train['clean_title'], test['clean_title'], "title")
    # get numeric features from
    logger.info("generating numeric features")

    train_num, test_num = numeric_features(train, test)

    logger.info("generating text and numeric features")
    # combing all features
    x_train = csr_matrix(hstack([train_tfidf_body, train_tfidf_title, train_num]))

    x_test = csr_matrix(hstack([test_tfidf_body, test_tfidf_title, test_num]))

    return x_train, x_test


def create_model(x_train, y_train, x_test, y_test):
    logger = logging.getLogger("train.create_model")
    logger.info("Training the model")
    #model = RandomForestClassifier(bootstrap=True, max_depth=70, random_state=0, n_estimators=400, min_samples_split=10,
    #                               min_samples_leaf=4)
    #model = RandomForestClassifier( random_state=0)
    model = AdaBoostClassifier(random_state=1)
    model.fit(x_train, y_train)
    logger.info("The accuracy of model on test set")
    print(model.score(x_test, y_test))
    return model


def evaluate_model(model, x_test, y_test, verbose=True):
    y_pred = model.predict(x_test)
    if verbose:
        # calculate Accuracy
        print("Accuracy:", accuracy_score(y_test, y_pred))

        # calculate recall
        print("Recall:", recall_score(y_true=y_test, y_pred=y_pred))

        # calculate precision
        print("Precision: ", precision_score(y_true=y_test, y_pred=y_pred))

        # calculate hamming loss
        print("Hamming Loss (%): ", hamming_loss(y_pred, y_test) * 100)

        # calculate F1 score
        print("F1 Score: ", f1_score(y_pred, y_test, average='weighted'))


def train(args):
    logger = logging.getLogger("train.train")
    input_data_fp, model_output_dir = args[1], args[2]

    logger.info("cleaning data and extracting features from text ")
    features = load(input_data_fp)
    # features.to_csv("/Users/nikhandelwal/stack-overflow-classifier-okc/data/clean_dataset_"+str(random.randint(0,1000))+".csv")

    logger.info("spliting data into a 80-20 split")
    x_train, x_test, y_train, y_test = train_test_split(features.drop('label', axis=1), features['label'],
                                                        test_size=0.2,
                                                        random_state=40)
    logger.info("generating training and testing data ")
    x_train, x_test = transform_features(x_train, x_test)
    logger.info("creating a random forest classifier ")
    model = create_model(x_train, y_train, x_test, y_test)
    logger.info("evaluating the model ")
    # evaluate_model(model, y_test, True)
    logger.info("saving model to disk a random forest classifier ")
    # save the model to disk
    filename = '/so_model.sav'
    logger.info("saving model to disk a random forest classifier " + filename)
    pickle.dump(model, open(model_output_dir + '/' + filename, 'wb'))


if __name__ == '__main__':
    train(sys.argv)
