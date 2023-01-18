import argparse
from argparse import Namespace
import nltk
import torch.cuda
from nltk.corpus import stopwords
from string import punctuation
nltk.download('stopwords')


def set_configuration() -> Namespace:
    """
    Function is used to collect parameters from the user and creates parser
    :return: Namespace object that contains all required information for the project
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, default='../datasets', required=False,
                        help='Specifies the dataset directory')
    parser.add_argument('--clean_stops', default=False, required=False, action='store_true',
                        help='Specifies whether stopwords will be in set (False) or not (True)')
    parser.add_argument('--clean_punctuation', default=False, required=False, action='store_true',
                        help='Specifies whether punctuation will be in set (False) or not (True)')
    parser.add_argument('--cased', default=False, required=False, action='store_true',
                        help='Specifies whether model will be case_sensitive (True) or not (False)')
    parser.add_argument('--window_size', default=200, type=int, required=False,
                        help='Specifies number of data a window can contain')
    parser.add_argument('--window_shift', default=200, type=int, required=False,
                        help='Specifies amount of steps a window will be shifted')
    parser.add_argument('--batch_size', default=128, type=int, required=False,
                        help='Specifies number of data will be kept in batch')
    parser.add_argument('--init_eval', required=False, action='store_true', default=False,
                        help='Specifies whether model should be evaluated before training starts (True) or not (False)')
    parser.add_argument('--experiment_number', required=False, type=int, default=13,
                        help='Specifies experiment number that user want to execute')
    parser.add_argument('--embedding_dimension', required=False, type=int, default=300,
                        help='Specifies embedding dimension of the LSTM model')
    parser.add_argument('--hidden_dimension', required=False, type=int, default=128,
                        help='Specifies hidden dimension of LSTM layer')
    parser.add_argument('--dropout', required=False, type=float, default=0.3,
                        help='Specifies dropout rate during the training')
    parser.add_argument('--bidirectional', required=False, action='store_true', default=False,
                        help='Specifies Bi-LSTM (True) or LSTM (False) will be used')
    parser.add_argument('--train_model', required=False, action='store_true', default=False,
                        help='Specifies whether model will be trained (True) or not (False)')
    parser.add_argument('--playground', required=False, action='store_true', default=False,
                        help='Specifies whether user wants to use the model to play around the data')
    parser.add_argument('--resume_training', required=False, action='store_true', default=False,
                        help='Specifies whether user wants to continue to train the model from the recent epoch or not')

    parser.add_argument('--optimizer', required=False, type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='Specifies optimizer choice of the user that can either be SGD or Adam')

    parser.add_argument('--learning_rate', required=False, type=float, default=0.002,
                        help='Learning rate for the model')
    parser.add_argument('--epochs', required=False, type=int, default=10,
                        help='Specifies number of epochs to train the model')
    parser.add_argument('--init_statistics', required=False, action='store_true', default=False,
                        help='Specifies whether initial information about data statistics should be provided or not')
    parser.add_argument('--set_length', required=False, action='store_true', default=False,
                        help='Specifies whether to use data driven length or not')
    parser.add_argument('--set_max_length', required=False, type=int, default=250,
                        help='Choice of max length that is done by user')

    parser.add_argument('--num_lstm_layers', required=False, type=int, default=3,
                        help='Specifies number of LSTM layers in the base model')

    parser.add_argument('--linear_dimensions', required=False, type=int, default=[100, 30], nargs='+',
                        help='Specifies list of hidden dimensions of FCN')
    parser.add_argument('--drop_list', required=False, type=float, default=[0.3, 0.2], nargs='+',
                        help='Specifies list of dropout rates')
    parser.add_argument('--play_bis', required=False, action='store_true', default=False,
                        help='Specifies whether word tokenization by done with BIS Model (True) or NLTK (False)')

    parser.add_argument('--playground_only', required=False, action='store_true', default=False,
                        help='Specifies whether only playground will be activated or not')

    return parser.parse_args()


def collect_parameters() -> dict:
    """
    Function collects user-defined parameters and adds initial parameters that are vital for the project
    :return: dictionary that contains all required information for the project
    """
    params_namespace = set_configuration()
    parameters = dict()
    for argument in vars(params_namespace):
        parameters[argument] = getattr(params_namespace, argument)
    parameters['punctuation'] = punctuation
    parameters['stopwords'] = stopwords.words('english')
    parameters['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return parameters
