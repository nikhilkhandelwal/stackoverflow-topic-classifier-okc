import os

import pandas as pd

import src.read.base_reader as reader

"""
Reading message log file and extract text or put them in pandas data frame
"""


class PandasRead(reader.Read):

    def __init__(self, filepath, file_type="csv", sep=",", header=0):
        """
        :param filepath:
        :param type: optional filetype, to keep track of different types of files
        :param sep: optional separator, to extract columns
        :param header: optional header, from the file
        """
        self._types = {"csv": self._read_csv, "txt": self._read_csv}
        self._filepath = filepath
        self._type = file_type
        self._sep = sep
        self._header = header
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                "Most likely you passed the wrong file path: {filepath} while creating Read instance".format(
                    filepath=filepath))
        assert file_type in self._types
        self._df = self._types[self._type](self._filepath, self._sep, self._header)

    def data_frame(self):
        """
        :return: dataframe which was populated when this class initialized
        """
        return self._df

    def text(self, text_column):
        """
        :param text_column: of the dataframe
        :return: array of the text messages
        """
        df = self._df
        if text_column in df.columns:
            return "\n".join(df[text_column].values)

    def _read_csv(self, filepath, sep, header):
        """
        Helper private function to read to dataframe from csv
        :param filepath:
        :param sep:
        :param header:
        :return: pandas dataframe
        """
        return pd.read_csv(filepath, sep=sep, header=header)
