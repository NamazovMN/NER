from torch.utils.data import Dataset
from read_process import ReadDataset
from tqdm import tqdm
import torch


class NERDataset(Dataset):
    """
    Class is dataset object, so that data processing will be initialized and data will be collected here according to
    the given dataset type
    """
    def __init__(self, config_parameters: dict, dataset_type: str):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_parameters: configuration parameters include all relevant data for the process
        :param dataset_type: type of the dataset that can either be training, dev or test
        """
        self.dataset_type = dataset_type
        self.process_obj = ReadDataset(config_parameters, self.dataset_type)
        self.data, self.label = self.get_dataset()

    def get_dataset(self) -> tuple:
        """
        Method is used to collect data for required form of Dataset
        :return: tuple that contains data and label tensors
        """
        data = list()
        label = list()
        ti = tqdm(iterable=self.process_obj, total=self.process_obj.__len__(),
                  desc=f'{self.dataset_type.title()} Dataset is prepared: ')
        for each_data, each_label in ti:
            data.extend(each_data)
            label.extend(each_label)

        return torch.LongTensor(data), torch.LongTensor(label)

    def __getitem__(self, item: int) -> dict:
        """
        Method gets index of data and return corresponding data and label sequences
        :param item: index of data
        :return: dictionary that includes data and label
        """
        return {
            'data': self.data[item],
            'label': self.label[item]
        }

    def __len__(self) -> int:
        """
        Function return length of dataset according to the processed data properties
        :return: length of the required dataset
        """
        return len(self.data)
