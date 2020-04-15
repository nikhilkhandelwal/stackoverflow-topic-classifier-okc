import pandas as pd
import src.read.pandas_reader as pr
import src.clean.title_cleaner  as tc
import src.clean.question_cleaner  as qc
import src.features.question_features  as qf
import src.features.title_features as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
count = 0


def load(filepath='/Users/nikhandelwal/stack-overflow-classifier-okc/data/small_interview_dataset.csv', training=True):
    """
    Read the CSV file and load it to pandas dataframe and apply cleaning and transformation function.
    Then we extract simple features and sentiments from the cleaned text tokens

    :param filepath: filepath of CSV file to load and extract features from
    :return: pandas datafeames of message_id, message_text, features, urgency as label
    :raises TypeError: if the filepath is not a string
    :raises FileNotFoundError: if the filepath does not exists
    """
    logger = logging.getLogger("utils.load")
    logger.info("loading dataframe")
    df = pr.PandasRead(filepath).data_frame()
    # df.drop(['Title_processed', 'Body_processed', 'Unnamed: 0'], axis=1, inplace=True)
    logger.info("cleaning data")

    df["clean_title_tokens"], df['clean_title'] = zip(*df["Title"].apply(question_title_cleaner))

    df["clean_body_tokens"], df['clean_body'] = zip(*df["Body"].apply(question_body_cleaner))

    logger.info("adding code block features ")

    df["code_block_len"], df['nums_code_block'], df['code_block'], df["question_len"], \
    df["url_count"], df['so_url_count'] = zip(
        *df[["clean_body", "clean_body_tokens", "Body"]].apply(get_question_features, axis=1))

    logger.info("adding title features ")

    df["title_len"] = df[["clean_title", "clean_title_tokens"]].apply(get_title_features, axis=1)

    # Returning dataframes separately for easier access later
    return df


def question_body_cleaner(text):
    cc = qc.QuestionCleaner(text.lower())
    cc = cc.clean()
    return cc.tokens(), cc.text()


def question_title_cleaner(text):
    cc = tc.TitleCleaner(text.lower())
    cc = cc.clean()
    return cc.tokens(), cc.text()


def get_question_features(r):
    """
    Function to extract features from text and cleaned tokens of text

    :param r: containing tuple of text and tokens
    :return: tuple of all the features
    """
    global count
    count += 1
    logger = logging.getLogger("utils.get_question_features")
    if count % 1000 == 0:
        logger.info(str(count) + " rows processed")
    text, tokens, original_text = r
    f = qf.QuestionFeatures(text, tokens, original_text)
    code_block, code_block_len, num_code_blocks = f.code_block_features()
    return code_block_len, num_code_blocks, code_block, f.question_len(), f.url_count(), f.so_url_count()


def get_title_features(r):
    """
    Function to extract features from text and cleaned tokens of text

    :param r: containing tuple of text and tokens
    :return: tuple of all the features
    """
    text, tokens = r
    f = tf.TitleFeatures(text, tokens)
    title_len = f.title_len()
    return title_len


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted
                                )
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted
                          )

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1
