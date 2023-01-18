import json
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from after_train import Computations
from auto_train import Auto_Train
from build_vocabulary import Prepare_Vocabulary
from build_dataset import Dataset_Generator
from datareader import Data_Reader
from lstm_ner import LSTM_NER
from os import path
from training import Training

from torch.utils.data import DataLoader

# Here we initialize data directories that will be used for processing the data
dataset_path = "../../data"
train_file = "train.tsv"
dev_file = "dev.tsv"
test_file = "test.tsv"
path_figure = "..//figures"
path_model = "..//models"
path_files = "..//files"
generated_data_path = "..//generated_data"
train_dataset_file = "trainset"
dev_dataset_file = "devset"
test_dataset_file = "testset"
vocabulary_file = "vocabulary.txt"
label_vocabulary_file = "label_vocabulary.txt"
# Here we define window_size, window_shift and device
device = "cuda"
window_size = 200
window_shift = 200
batch_size = 128


def write_data(path_data, file_name, data):
    file_path = os.path.join(path_data, file_name)
    with open(file_path, 'w') as data_file:
        data_file.write(json.dumps(data))
    print("{} was generated and data was saved successfully!".format(file_name))


def load_data(path_data, file_name):
    file_path = os.path.join(path_data, file_name)
    data = {}
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def save_datasets(path_data, file_name, data):
    file_path = os.path.join(path_data, file_name)
    data_np = np.array(data)
    np.save(file_path, data_np, allow_pickle=True)
    print("{} was generated successfully!".format(file_name))


def load_datasets(path_data, file_name):
    file_name = file_name + ".npy"
    file_path = os.path.join(path_data, file_name)
    dataset_np = np.load(file_path, allow_pickle=True)
    return list(dataset_np)


# Declaration of classes according to the data files

## DATA READER
### data_reader for training data
dr_train = Data_Reader(dataset_path, train_file)
### data_reader for development data
dr_dev = Data_Reader(dataset_path, dev_file)
### data_reader for training data
dr_test = Data_Reader(dataset_path, test_file)

## VOCABULARY BUILDING
### vocabulary_builder will be used according to only the train data
vocabulary_builder = Prepare_Vocabulary(dr_train, train_file)

## DATASET BUILDING
### dataset_builder for train data
dg_train = Dataset_Generator(dr_train, vocabulary_builder, device, window_size, window_shift)
### dataset_builder for development data
dg_dev = Dataset_Generator(dr_dev, vocabulary_builder, device, window_size, window_shift)
### dataset_builder for test data
dg_test = Dataset_Generator(dr_test, vocabulary_builder, device, window_size, window_shift)

trainingset = []
devset = []
testset = []
vocabulary = {}
label_vocabulary = {}
#################---------------################--------------############
if not os.listdir(generated_data_path):
    print("Generated data directory is empty! That is why data will be generated!")
    vocabulary, label_vocabulary = vocabulary_builder.generate_vocabulary()

    # save vocabularies to text file
    write_data(generated_data_path, vocabulary_file, vocabulary)
    write_data(generated_data_path, label_vocabulary_file, label_vocabulary)

    #################---------------################--------------############
    # Here we build dataset for model

    trainingset = dg_train.encode_dataset()
    devset = dg_test.encode_dataset()
    testset = dg_test.encode_dataset()

    # save datasets
    save_datasets(generated_data_path, train_dataset_file, trainingset)
    save_datasets(generated_data_path, dev_dataset_file, devset)
    save_datasets(generated_data_path, test_dataset_file, testset)

    # Getting batched in datasets
else:
    print("Data is loaded ...")
    print(40 * "*")

    trainingset = load_datasets(generated_data_path, train_dataset_file)
    devset = load_datasets(generated_data_path, dev_dataset_file)
    testset = load_datasets(generated_data_path, test_dataset_file)
    vocabulary = load_data(generated_data_path, vocabulary_file)
    label_vocabulary = load_data(generated_data_path, label_vocabulary_file)

train_dataset = DataLoader(trainingset, batch_size=batch_size)
dev_dataset = DataLoader(devset, batch_size=batch_size)
test_dataset = DataLoader(testset, batch_size=batch_size)

# checking length of datasets and batched datasets

print("training dataset has {} sentences".format(len(trainingset)))
print("development dataset has {} sentences".format(len(devset)))
print("test dataset has {} sentences".format(len(testset)))
print(40 * "*")
print("training dataset has {} batches".format(len(train_dataset)))
print("development dataset has {} batches".format(len(dev_dataset)))
print("test dataset has {} batches".format(len(test_dataset)))

print(40 * "*")
print(40 * "*")

# Building model and training phase

## Model variables
output_size = len(label_vocabulary)
embedding_dim = 100
hidden_dim = 128
n_layers = 2

## Training Phase parameters
learning_rate = 0.002
epochs = 20
loss_function = nn.CrossEntropyLoss(ignore_index=label_vocabulary["<pad>"])

a_t = Auto_Train(vocabulary, label_vocabulary, train_dataset, dev_dataset, test_dataset, loss_function, output_size,
                 hidden_dim, embedding_dim, n_layers, path_model, path_figure, path_files)

lr = [0.001, 0.002, 0.005, 0.01]
lr_name = ["0_001", "0_002", "0_005", "0_01"]
optimizers = ["Adam"]
models = [True, False]

a_t.train_auto(models, optimizers, lr, lr_name, epochs)
