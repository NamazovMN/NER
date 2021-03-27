import numpy as np
import os
import pandas as pd
from datareader import Data_Reader
from build_vocabulary import Prepare_Vocabulary
import torch
from torch.utils.data import DataLoader
class Dataset_Generator(object):
    def __init__(self, data_reader, vocab_builder, device, window_size, window_shift):
        self.vocab_builder = vocab_builder
        self.data_reader = data_reader
        self.device = device
        self.window_size = window_size
        self.window_shift = window_shift

    
    def get_sentences(self):
        dr = self.data_reader.iterate_data_file()
        sentences_data = []
        sentences_labels = []
        for each_sentence in dr:
            data = []
            labels = []
            for each_token in each_sentence:
                data.append(each_token["form"])
                labels.append(each_token["lemma"])
            sentences_data.append(data)
            sentences_labels.append(labels)
        return sentences_data, sentences_labels
    
    def generate_windows(self, sentences):
        # data_vocabulary, labels_vocabulary = self.vocab_builder.generate_vocabulary()
        # data_sentences, data_labels = self.get_sentences()
        data = []
        for each in sentences:
            for i in range(0, len(each), self.window_shift):
                window = each[i:i+self.window_size]
                if len(window)<self.window_size:
                    window += [None]*(self.window_size-len(window))
            data.append(window)
        return data

    def encode_text(self, sentence, vocabulary):

        encoded_sentence = []
        for each in sentence:
            if each is None: # padding encoding is done here
                encoded_sentence.append(vocabulary["<pad>"])
            elif each not in vocabulary.keys(): # this elif will be used only for data, because there will not be such label that is not in our labels set
                encoded_sentence.append(vocabulary["<unk>"])
            else: # encoding normal data according to the vocabulary, labels according to the labels set
                encoded_sentence.append(vocabulary[each])
        
        return encoded_sentence



    def encode_dataset(self):
        # here we generate vocabulary at once and we use it when we need. 
        data_vocabulary, labels_vocabulary = self.vocab_builder.generate_vocabulary()
        # we get raw sentences and labels (according to tokens)
        data_sentences, labels_sentences = self.get_sentences()
        # generate windows according to window size for data and labels
        data_windows = self.generate_windows(data_sentences)
        labels_windows = self.generate_windows(labels_sentences)
        encoded_dataset = []
        # we encode data with that for loop by using encode text function
        for each_data, each_label in zip(data_windows, labels_windows):
            encoded_dataset.append({"input_data":torch.LongTensor(self.encode_text(each_data, data_vocabulary)),
                                    "output_data":torch.LongTensor(self.encode_text(each_label, labels_vocabulary))})

        return encoded_dataset


    def reverse_vocabulary(self, vocabulary):
        """
        Args: 
            vocabulary: dictionary such that keys are NER tags, values are indexes

        output: 
            reverse_vocabulary: dictionary such that keys are indexes, values are NER tags

        """
        reverse_vocab = {}
        for each in vocabulary.keys():
            reverse_vocab[vocabulary[each]] = vocabulary[each]
        return reverse_vocab


    def decode_dataset(self, output, label_vocabulary):
        """
        Args: 
            vocabulary: dictionary such that keys are NER tags, values are indexes
            output: output of the model in type of Long Tensor which shape is (batch_size, max_len (in our case 200), num_of_classes)
        output: 
            prediction: list of lists. Each list is the NER tags for sentences in one batch.

        """
        
        max_vals = torch.argmax(output,-1).tolist()
        l_vocab = self.reverse_vocabulary(label_vocabulary)
        prediction = []
        for each_idx_of_batch in max_vals:
            prediction_batch = list()
            for each_idx in each_idx_of_batch:
                prediction_batch.append(l_vocab[each_idx])
            prediction.append(prediction_batch)
        return prediction





# dr = Data_Reader("..//data")
# bv = Prepare_Vocabulary(dr, "train.tsv")
# dg = Dataset_Generator("train.tsv", dr, bv,'cuda',200,200)

# ed= dg.encode_dataset()

# td = DataLoader(ed, batch_size = 64)
# print(len(td))

# print(ed[0])
