import os 
import csv
import numpy
import tqdm
import pandas as pd

from conllu import parse_incr


class Data_Reader(object):
    def __init__(self, dataset_path, filename):
        self.path = dataset_path
        self.filename = filename
    
    def generate_dataset(self):
        return os.path.join(self.path, self.filename)

    def iterate_data_file(self):
        # this function generates conllu parser generator that we can iterate sentences over this iterator.
        # each sentences is generated as the "Tokenlist" which includes id (place of token in sentence), form (token itself),
        # lemma (entity recognition of token (i.e. labels))
        dataset = self.generate_dataset()
        data_file = open(dataset, "r", encoding="utf-8")
        return parse_incr(data_file)        

