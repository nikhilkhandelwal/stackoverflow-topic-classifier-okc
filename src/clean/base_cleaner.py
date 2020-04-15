import abc

"""
Abstract class to implement cleaner and three important functions remove, normalize and mask
"""


class Cleaner(abc.ABC):

    def __init__(self, text):
        self._text = text
        self._tokens = text.split(" ")  # Very basic tokenizer

    @abc.abstractmethod
    def normalize(self):
        return self

    def clean(self):
        """
        Calling functions in a specific order to get the best cleaning results
        :return: self
        """
        self.normalize()
        return self

    def text(self):
        """
        :return: cleaned text
        """
        return self._text

    def tokens(self):
        """
        :return: cleaned tokens
        """
        return self._tokens