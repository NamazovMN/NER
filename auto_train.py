import json
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from after_train import Computations
from build_vocabulary import Prepare_Vocabulary
from build_dataset import Dataset_Generator
from datareader import Data_Reader
from lstm_ner import LSTM_NER
from training import Training

from torch.utils.data import DataLoader


class Auto_Train(object):
    def __init__(self, vocabulary, label_vocabulary, train_dataset, dev_dataset, test_dataset, loss_function, output_size, hidden_dim, embedding_dim, n_layers, path_model, path_figure, path_files):
        self.vocabulary = vocabulary
        self.label_vocabulary = vocabulary
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.loss_function = loss_function
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.path_model = path_model
        self.path_figure = path_figure
        self.path_files = path_files

    def write_data(self, file_name, data):
        file_path = os.path.join(self.path_files, file_name)
        with open(file_path, 'w') as data_file:
            data_file.write(json.dumps(data))
            print("{} was generated and data saved!".format(file_name))
            print(40*"*")


    def train_auto(self, models, optimizers, lr,lr_name, epochs_init):
        losses_dictionary = {}
        train_losses = {}
        dev_losses = {}
        precisions_dict = {}
        f1_results = {}
        model_names = {}
        models_data =[]
        epochs = 0
        count = 0
        total_count = len(models)*len(optimizers)*len(lr)

        for each in models:
            for each_lr, each_name_lr in zip(lr, lr_name):
                count += 1
                model = LSTM_NER(self.vocabulary, len(self.vocabulary), self.output_size, self.embedding_dim, self.hidden_dim, self.n_layers, bidirectional=each).cuda()
                computations = Computations(model, self.vocabulary, self.label_vocabulary, self.path_figure, self.path_model )
                
                optimizer = optim.Adam(model.parameters(), each_lr)
                epochs = epochs_init
            
                if each == True:
                    model_name = "Bi-LSTM_Adam_"+each_name_lr
                else:
                    model_name = "LSTM_Adam_"+each_name_lr
            
                print("Training of {}/{} model ({}) started.".format(count,total_count, model_name))
                print(40*"*")

                trainer = Training(model, optimizer, self.loss_function, self.train_dataset, self.dev_dataset)
                _, train_loss_list, dev_loss_list = trainer.train_phase(epochs)
                train_losses[model_name] = train_loss_list
                dev_losses[model_name] = dev_loss_list
                precisions, predictions, labels = computations.compute_precision(self.test_dataset)
                computations.compute_confusion_matrix(predictions, labels, model_name)
                computations.generate_graphs(train_loss_list, dev_loss_list, epochs, model_name)
                f1_macro, f1_micro = computations.compute_f1_score(predictions, labels)

                f1_results[model_name] = {'macro':f1_macro,'micro':f1_micro}                
                precisions_dict[model_name] = {'macro':precisions['macro_precision'], 'micro': precisions['micro_precision']}
                models_data.append(model_name)
                model_path = os.path.join(self.path_model, model_name)
                torch.save(model.state_dict(), model_path)

                print("Micro Precision: {}\nMacro Precision: {}".format(precisions["micro_precision"], precisions["macro_precision"]))
                print("Micro F1: {}\nMacro F1: {}".format(f1_micro, f1_macro))

                print(40*"*")

        
        losses_dictionary = {"train":train_losses, "dev": dev_losses}
        model_names = {'models': models_data}
        self.write_data("losses.txt", losses_dictionary)
        self.write_data("precisions.txt", precisions_dict)
        self.write_data('f1_results.txt', f1_results)
        self.write_data('model_names.txt', model_names)
