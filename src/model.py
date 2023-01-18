import collections

import torch
from torch import nn


class BuildModelStructure:
    """
    Class is used to build dynamic model structure according to user specifications along with checking compatibility
    that user provides
    """
    def __init__(self, config_params):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_params: configuration parameters include all relevant data for the process
        """
        self.configuration = config_params
        self.model_base = self.create_model_structure()

    def get_lstm_out(self) -> int:
        """
        Method is used to define output size of the lstm layer according to the bidirectional parameter
        :return:
        """
        if self.configuration['bidirectional']:
            return self.configuration['hidden_dimension'] * 2
        else:
            return self.configuration['hidden_dimension']

    def lstm_layer(self) -> nn.LSTM:
        """
        Method is used to define LSTM layer according to the user's specifications
        :return: LSTM layer(s) for the classifier model
        """
        return nn.LSTM(self.configuration['embedding_dimension'], self.configuration['hidden_dimension'],
                       dropout=self.configuration['dropout'], bidirectional=self.configuration['bidirectional'],
                       num_layers=self.configuration['num_lstm_layers'])

    def embedding_layer(self) -> nn.Embedding:
        """
        Method is used to define embedding layer according to the user's specifications
        :return: Embedding layer for the classifier model
        """
        return nn.Embedding(self.configuration['vocabulary'].__len__, self.configuration['embedding_dimension'])

    def create_model_structure(self) -> dict:
        """
        Method is used to create model structure and returns the model layers
        :return: dictionary that specifies model's layers according to the user's specifications
        """
        if len(self.configuration['linear_dimensions']) != len(self.configuration['drop_list']):
            raise Exception('Number of linear layers and length of dropout list are not compatible! '
                            'Note: Output layer should not be added to both parameters')

        base_model = {'embedding': self.embedding_layer(), 'lstm': self.lstm_layer()}
        hidden = self.get_lstm_out()
        if len(self.configuration['linear_dimensions']):
            for idx, dimension in enumerate(self.configuration['linear_dimensions']):
                if idx == 0:
                    linear = nn.Linear(self.get_lstm_out(), dimension)
                else:
                    linear = nn.Linear(self.configuration['linear_dimensions'][idx - 1], dimension)
                base_model[f'linear_{idx}'] = linear
                base_model[f'dropout_{idx}'] = nn.Dropout(self.configuration['drop_list'][idx])
                base_model[f'relu_{idx}'] = nn.ReLU()
            hidden = self.configuration['linear_dimensions'][-1]
        base_model['output_layer'] = nn.Linear(hidden, len(self.configuration['vocabulary'].label_dict))
        return base_model

    def __iter__(self) -> collections.Iterable:
        """
        Method is used as default iterator of the class to iterate over layers of the object
        :yield: tuple of name of layer and layer itself
        """
        for name, module in self.model_base.items():
            yield name, module


class Classifier(nn.Module):
    """
    Class is an object for the classifier model, which layers are added dynamically according to the user's choices
    """
    def __init__(self, model_struct_object: BuildModelStructure):
        """
        Method initializes the class to set the configuration and initial variables
        :param model_struct_object: Model structure class that helps us to add layers to the model dynamically
        """
        super(Classifier, self).__init__()
        for name, module in model_struct_object:
            self.add_module(name, module)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Method is used for feedforward process
        :param input_data: Tensor data of corresponding sequence or batch of sequences
        :return: Tensor data of model's output for sequence or batch of sequences
        """
        for module in self.children():
            if type(module) == nn.LSTM:
                input_data, (_, _) = module(input_data)
            else:
                input_data = module(input_data)
        return input_data
