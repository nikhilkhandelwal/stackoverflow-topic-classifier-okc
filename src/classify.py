import sys
import json
import pandas as pd
from src.utils.utils import *
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import logging

model = None
import numpy as np

logging.basicConfig(level=logging.INFO)


def get_input_features(df_path):
    return load(df_path, False)


def load_model(model_path):
    return pickle.load(open(model_path, 'rb'))


def tfidf_features(model_path, text, type):
    vec_path = model_path + "/" + type + "_tfidf_vectorizer.pickle"
    vectorizer = pickle.load(open(vec_path, "rb"))
    return vectorizer.transform(text)


def save_results(predictions):
    output_fp = "output_file.csv"
    predictions = pd.DataFrame(predictions, columns=['pred_labels'])
    predictions.to_csv(output_fp)
    return output_fp


def classify(args):
    global model

    logger = logging.getLogger("classify.classify")
    model_path, df_path = args[1], args[2]
    logger.info("getting features for input data")
    # get basic features
    features = get_input_features(df_path)
    # transform text to tf-idf features
    title_tfidf = tfidf_features(model_path, features['clean_title'], "title")
    question_tfidf = tfidf_features(model_path, features['clean_body'], "question")

    numeric_features = features[
        ['title_len', 'question_len', 'code_block_len', 'nums_code_block', 'url_count', 'so_url_count']]
    # concatenate tfidf and numeric features
    data = csr_matrix(hstack([question_tfidf, title_tfidf, numeric_features]))
    logger.info("loading model")
    # check if model is loaded
    if not model:
        model = load_model(model_path + "/so_model.sav")
    # make prediction
    logger.info("getting predictions model")
    predictions = model.predict(data)

    logger.info("writing predictions to file")
    fp = save_results(predictions)
    logger.info("results written to file: " + fp)


if __name__ == '__main__':
    classify(sys.argv)
