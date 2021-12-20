import csv
import random
from collections import OrderedDict
from GSKhopt.CSVDataFrame.utils_numbers import isfloat
import pandas as pd
import itertools
class CSVDataFrame (object):
    def __init__(self,filename=None):
        self.filename = filename
        self.DataFrame = []
        self.header = {}
        self.rows = 0
        self.cols = 0
    def ReadCSV(self,header_exists=True):
        with open(self.filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if header_exists:
                    for c,col in enumerate(row):
                        self.header.update({col:c})
                    header_exists = False
                    self.header = OrderedDict(sorted(self.header.items(), key=lambda x: x[1]))
                    continue
                self.DataFrame.append(row)

        self.rows = len(self.DataFrame)
        if header_exists:
            self.cols = len(self.header)
        else:
            #assuming csv file
            self.cols = len(self.DataFrame[0])
    def setheader(self, header):
        for c,col in enumerate(header):
            self.header.update({col:c})
    def getheader(self):
        return header
    def PassDataFrame(self, DataFrame, header=None):
        #pass a new header
        self.DataFrame = DataFrame
        if header != None:
            for c,col in enumerate(header):
                self.header.update({col:c})
            self.header = OrderedDict(sorted(self.header.items(), key=lambda x: x[1]))

            self.rows = len(self.DataFrame)
            self.cols = len(self.header)
        else:
            self.rows = len(self.DataFrame)
            self.cols = len(self.DataFrame[0])

    def shuffle(self):
        random.shuffle(self.DataFrame)
    def sort(self, key=None):
        #sort the dataframe based on a key
        if key == None:
            return None
        else:
            self.DataFrame.sort(key=lambda x: x[self.header[key]])
    def head(self,n=1):
        if n > self.rows:
            n = self.rows
        print ('Showing ' + str(n) + ' rows x ' + str(self.cols) +'cols')
        print  ('index\t' + '\t'.join(self.header))
        print ('-----' * self.cols)
        for i in range(n):
            print (i , '\t' ,'\t'.join(self.DataFrame[i]))
    def concat(self,DataFrame2):
        if self.header  == DataFrame2.header:
            for row in DataFrame2.DataFrame:
                self.DataFrame.append(row)
            self.rows = len(self.DataFrame)
    def getrowindex(self,key=None,attr=None):
        for i, row in enumerate(self.DataFrame):
            if row[self.header[key]] == attr:
                return i

    def selectrows(self,keys=None,attrs= None):
        rows = []
        for i in range(self.rows):
            equal = True
            for k, attr in itertools.izip(keys,attrs):
                if self.DataFrame[i][self.header[k]] != attr:
                    equal = False
                    break
            if equal:
                rows.append(i)
        return rows




    def selectbycol(self,rows = None, keys=None):
        if rows == None:
            rows = self.rows
        return [[self.DataFrame[i][self.header[k]] for k in keys] for i in range(rows)]
    def seperatebycol(self,rows = None , keys = None):
        '''
        This was done by the following code sample:
        s = set(temp2)
        temp3 = [x for x in temp1 if x not in s]
        '''
        sk = set(keys)
        new_keys = [k for k in self.header.keys() if k not in sk]
        return self.selectbycol(rows,keys) , self.selectbycol(rows,new_keys)
    def convert2num(self):
        return [[float(val) if isfloat(val) else val for val in item] for item in self.DataFrame]
    def convert2pandasdf(self,dtype = object):
        return pd.DataFrame(self.DataFrame,columns=self.header.keys(),dtype=dtype)
    def save(self,new_name=None, no_header=False):
        self.header = OrderedDict(sorted(self.header.items(), key=lambda x: x[1]))
        if no_header:
            data = self.DataFrame
        else:
            data = [self.header.keys()]
            data += self.DataFrame
        if new_name == None:
            file = open(self.filename, 'w')
        else:
            file = open(new_name, 'w')
        with file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(data)
