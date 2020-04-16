import src.features.base_feature as bf
from bs4 import BeautifulSoup
import re

"""
Extract features from question using tokens and text
All the functions are self explanatory  
"""


class QuestionFeatures(bf.Features):

    def __init__(self, text, tokens, original_text):
        self._text = text
        self._tokens = tokens
        self._original_text = original_text

    def code_block_features(self):
        soup = BeautifulSoup(self._original_text, 'html.parser')
        res = list()
        total_len = 0
        code_block = ''
        for a in soup.find_all('pre'):
            res.append(a.string)
            if a.string:
                total_len += len(a.string)
        if res:
            for s in res:
                if s:
                    code_block += ''.join(s) + " "
        return code_block, total_len, len(res)

    def url_count(self):
        # url counts
        urls = re.findall(r'(https?://\S+)', self._original_text)
        return len(urls)

    def so_url_count(self):
        # stack over flow url counts
        urls = re.findall(r'(https?://stackoverflow.com\S+)', self._original_text)
        return len(urls)

    def question_len(self):
        return len(self._tokens)
