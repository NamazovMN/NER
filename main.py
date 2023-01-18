from torch.utils.data import DataLoader
from model import Classifier, BuildModelStructure
from dataset import NERDataset
from utilities import *
from vocab import Vocabulary
from trainer import Train
from statistics import Statistics
import os
import pickle
from read_process import ReadDataset
from playground import Playground


def get_loaders(config_parameters: dict):
    """
    Function is used to collect data loaders for all dataset types: train, dev, test
    :param config_parameters: initial configuration parameters for the project
    :return: dictionary that contains all data loaders
    """
    dataset_types = ['train', 'dev', 'test']
    data_loaders = dict()
    for each_type in dataset_types:
        ds = NERDataset(config_parameters, each_type)
        data_loaders[each_type] = DataLoader(
            ds,
            batch_size=config_parameters['batch_size'],
            shuffle=True
        )
    return data_loaders

def get_max_length(config_parameters: dict) -> int:
    """
    Function is used to get maximum sequence length according to the train data. Dev data is not the one that model is
    trained on. That is why, we consider only train dataset
    :param config_parameters: initial configuration parameters for the project
    :return: maximum sequence length in the train dataset
    """
    config_path = os.path.join('train_results', f"experiment_{config_parameters['experiment_number']}")

    if config_parameters['playground_only']:
        if not os.path.exists(config_path):
            raise FileNotFoundError('Check the train results path to find correct experiment number!')
        config_file = os.path.join(config_path, 'model_config.pickle')
        with open(config_file, 'rb') as config_data:
            config_params = pickle.load(config_data)
        max_length = config_params['max_length']
    else:
        train_reader = ReadDataset(config_parameters, 'train', length_check=True)

        train_ds = train_reader.read_data()
        lengths = train_ds['length']
        max_length = max(lengths)
    return max_length


def __main__() -> None:
    """
    Main function which collects all required parameters and organizes whole procedure for the task
    :return None
    """
    parameters = collect_parameters()
    parameters['vocabulary'] = Vocabulary(parameters)
    parameters['max_length'] = get_max_length(parameters)

    bms = BuildModelStructure(parameters)
    classifier = Classifier(bms).to(parameters['device'])

    if not parameters['playground_only']:
        dataloaders = get_loaders(parameters)
        trainer = Train(parameters, classifier)
        parameters['experiment_environment'] = trainer.configuration['environment']
        parameters['model_structure'] = bms

        statistics = Statistics(parameters)
        max_label = statistics.provide_statistics(before_training=True)
        if parameters['train_model']:
            trainer.train_epoch(dataloaders, max_label)
            statistics.provide_statistics(before_training=False)

    playground_obj = Playground(parameters, classifier)
    playground_obj.process()

if __name__ == '__main__':
    __main__()

