import csv
import json
import yaml
import pickle as pkl
import numpy as np

from pathlib import Path

class Logger:
    supported_fmts = []

'''Thoughts
the data should all be streamed, row by row if possible, since the experiments could get quite large.
this is obviously a problem, e.g. for pickle, json etc.
what we could do is write to a redis cache keyed on the experiment name and then dump?
let's only do csv for now

'''
class Data:
    supported_fmts = [] # e.g. ['csv','json']
    def __init__(self):
        self.x = None
        self.y = None
        self.writers = {
            'csv': self.write_csv
        }
    def add_writer(self, fmt, write_func):
        if not self.check_supported_fmts(fmt):
            raise ValueError("Trying to add a writer for an unsupported format {}!".format(fmt))
        if fmt in self.writers:
            print("Overwriting write function for: {}".format(fmt))
        self.writers[fmt] = write_func


    def check_supported_fmt(fmt):
        '''Check if a format is supported by this data class 
            (for reading or writing to file)
            Returns True if supported, False if not.
        '''
        if fmt in supported_fmts:
            return True
        else:
            return False

    @staticmethod
    def infer_fmt(fname):
        '''Get the extension for a filename/path
        '''
        return Path(fname).suffix[1:] # strip out the dot


    def write_data(self,fname,fmt):
        '''Write all the data in memory to a file
        '''
        raise NotImplementedError
    def load_data(self,fname,fmt=None):
        '''Loads a datafile and parses it into a Data object
        '''
        raise NotImplementedError
    def write_entry(self,entry):
        '''Write a single entry to file without saving it in memory
        '''
        raise NotImplementedError
    def as_np_array(self):
        raise NotImplementedError
    def as_dataframe(self):
        raise NotImplementedError
    def write_csv(self):
        print("hi csv!")

    def load_pickle(self,fname):
        with open(fname, 'rb') as infile:
            return pkl.load(infile)
    def load_csv(self, fname, np_arr=True):
        with open(fname, 'r') as infile:
            reader = csv.reader(infile,quoting=csv.QUOTE_NONNUMERIC) #convert non-quoted to floats
            data = []
            for line in reader:
                data.append(line)
            if np_arr:
                return np.array(data)
            else:
                return data
    def load_json(self,fname):
        with open(fname) as jfile:
            return json.load(jfile)


class ClassificationData(Data):
    supported_fmts = ['csv']
    def to_file(self,fname,fmt):
        if self.check_supported_fmt(fmt):
            raise(ValueError("Trying to write to unsupported format: {}".format(fmt)))
        if fmt == 'csv':
            with open(fname, 'w') as outfile:
                data_writer = csv.writer(outfile, delimiter=',')
                for row in self.data:
                    data_writer.writerow(row)
    def load_data(self,fname,fmt=None):
        if fmt is None:
            # try to infer the format from the filename
            fmt = self.infer_fmt(fname)
    def write_csv(self):
        print('bye csv!')

# add any data_types you write to this list and __all__
import data_types.distopia_data
__all__ = ['distopia_data']
