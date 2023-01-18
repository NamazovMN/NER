from torch import nn
import torch


class BISClassifier(nn.Module):
    """
    Class is the model object to perform the BIS classifier task
    """

    def __init__(self, config_params):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_params: configuration parameters include all relevant data for the process
        """
        super(BISClassifier, self).__init__()

        self.embedding = nn.Embedding(len(config_params['vocabulary']), config_params['embedding_dimension'])
        self.lstm = nn.LSTM(config_params['embedding_dimension'], config_params['hidden_dimension'],
                            bidirectional=config_params['bidirectional'], num_layers=2,
                            dropout=config_params['dropout'])
        lstm_out_dim = 2 * config_params['hidden_dimension'] if config_params['bidirectional'] \
            else config_params['hidden_dimension']
        self.linear = nn.Linear(lstm_out_dim, config_params['lin_out'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config_params['dropout'])
        self.out = nn.Linear(config_params['lin_out'], 4)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Method performs feed-forward process of the model
        :param input_data: torch tensor of the input data
        :return: Output of the model for the provided input data
        """
        h_embedding = self.embedding(input_data)
        h_lstm, (_, _) = self.lstm(h_embedding)
        linear_out = self.relu(self.dropout(self.linear(h_lstm)))
        out = self.out(linear_out)

        return out
