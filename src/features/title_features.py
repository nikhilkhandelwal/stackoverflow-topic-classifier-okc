import src.features.base_feature as bf
from bs4 import BeautifulSoup

"""
Extract features from messages using tokens and text
All the functions are self explanatory  
"""


class TitleFeatures(bf.Features):
    def __init__(self, text, tokens):
        self._text = text
        self._tokens = tokens

    def title_len(self):

        return len(self._text)

