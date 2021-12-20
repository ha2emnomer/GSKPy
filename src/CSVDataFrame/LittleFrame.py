from collections import OrderedDict
import pandas as pd
class CSVDataFrame (object):
    def __init__(self, data):
        self.filename = filename
        self.DataFrame = data
        self.rows = 0
        self.cols = 0
    def __sizeof__(self):
        return self.rows
