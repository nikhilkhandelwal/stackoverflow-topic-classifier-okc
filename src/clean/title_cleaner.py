import re
from abc import ABC
import nltk.tokenize as tokenizer
import stop_words as stop_words
from nltk.corpus import stopwords
import nltk.tokenize as tokenizer
from nltk.stem.snowball import SnowballStemmer
import src.clean.base_cleaner as clean

stemmer = SnowballStemmer('english')

"""
Question Cleaner for cleaning stack over questions. All function names are self explanatory.
Ideally if I had more time I would have made more elaborate subroutines, for each cleaning step a separate function 
"""


# Preprocess the question title for vectorization
# - Convert to lowercase
# - Remove stopwords
# - Remove HTML
# - Remove special characters
# - Stemming


class TitleCleaner(clean.Cleaner, ABC):
    def __init__(self, text):
        self._tk = tokenizer.SpaceTokenizer()
        self._text = text
        self._tokens = self._tk.tokenize(self._text)

        # Some constants which ideally should be loaded from a config file or a model
        self._STOP_WORDS = stop_words.get_stop_words('en')

    def _assign(self, txt):
        """
        After each step do the following steps to clean up the mess created in those functions
        Change tokens and text of self
        :param txt:
        :return: None
        """
        txt = re.sub("\s+", " ", txt)
        self._text = txt.strip()
        self._tokens = self._tk.tokenize(self._text)

    def _map(self, frm, to, txt):
        """
        maps a word/ phrase with a masking phrase, finds text in the beginning, mid and end
        of the sentence with space around it
        :param frm:
        :param to:
        :param txt:
        :return: text after mapping
        """
        frm = r"^{frm} | {frm} | {frm}$".format(frm=frm)
        return re.sub(frm, to, txt)

    def normalize(self):
        """
        :return: self
        """
        text = self._text
        self._assign(text)
        text = self._remove_stopwords()
        self._assign(text)
        text = self._remove_punc()
        self._assign(text)
        text = self._stem_text()
        self._assign(text)
        return self

    def _remove_html(self):
        """
        :return: self
        """
        text = self._text
        # Remove html
        return re.sub(r"\<[^\>]\>", "", text)

    def _remove_stopwords(self):
        """
        :return: self
        """
        text = self._text
        # tokenize the text
        tokens = self._tokens

        filtered = [w for w in tokens if not w in self._STOP_WORDS]
        return ' '.join(map(str, filtered))

    def _remove_punc(self):
        """
        :return: self
        """
        text = self._text
        # tokenize
        tokens = self._tokens

        # remove punctuations from each token
        tokens = list(map(lambda token: re.sub(r"[^A-Za-z0-9]+", " ", token).strip(), tokens))

        # remove empty strings from tokens
        tokens = list(filter(lambda token: token, tokens))

        return ' '.join(map(str, tokens))

    def _stem_text(self):
        """
        :return: self
        """

        tokens = self._tokens
        # stem each token
        tokens = list(map(lambda token: stemmer.stem(token), tokens))

        return ' '.join(map(str, tokens))

    def __repr__(self):
        return self._text
