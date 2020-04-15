import abc

"""
Abstract class to implement readers for different types of files in more processable format 
"""


class Read(abc.ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def text(self, *args, **kwargs):
        pass
