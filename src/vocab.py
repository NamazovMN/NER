import os
import pickle
from tqdm import tqdm
from collections import Counter


class Vocabulary:
    """
    Class is used as a Vocabulary object. It will be utilized for encoding the input data and decoding the output data
    """
    def __init__(self, config_parameters: dict, is_bis: bool = False):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_parameters: initial configuration parameters to set the vocabulary
        :param is_bis: boolean variable which specifies whether vocabulary is defined for BIS or NER
        """
        self.configuration = self.set_configuration(config_parameters)
        self.vocabulary, self.label_dict = self.create_bis_vocabulary() if is_bis else self.create_vocabulary()
        self.id2label = self.reverse_labels()

    @staticmethod
    def check_dir(directory: str) -> str:
        """
        Method is used to check whether the given path exists or not. If it does not exist, method creates and
        returns the directory
        :param directory: path that given to check its existence
        :return: existent path
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def set_configuration(self, parameters):
        """
        Method is used to extract required parameters and return specific parameters dictionary
        :param parameters: configuration parameters dictionary with all parameters for the task
        :return: dictionary with required parameters
        """
        source_file = os.path.join(parameters['dataset_dir'], 'data/train.tsv')
        processed_folder = 'processed'
        if parameters['clean_stops']:
            processed_folder += '_s'
        if parameters['clean_punctuation']:
            processed_folder += '_p'
        if parameters['cased']:
            processed_folder += '_cased'
        processed_folder = self.check_dir(os.path.join(parameters['dataset_dir'], processed_folder))
        return {
            'dataset_dir': parameters['dataset_dir'],
            'source_data': source_file,
            'processed_folder': processed_folder,
            'clean_stops': parameters['clean_stops'],
            'clean_punctuation': parameters['clean_punctuation'],
            'stopwords': parameters['stopwords'],
            'punctuation': parameters['punctuation'],
            'cased': parameters['cased']
        }

    def check_validity(self, token: str) -> tuple:
        """
        Method is used to check token's validity for the given scenario by user, such as:
            cased: if tokens will be case-sensitive (True) or not (False);
            clean_stops: if stopwords will be eliminated (True) or not (False);
            clean_punctuation: if punctuations will be eliminated (True) or not (False)
        :param token: character which validity is needed to be checked
        :return: tuple object which includes token itself and its validation result as boolean variable
        """
        result = True
        if not self.configuration['cased']:
            token = token.lower()
        if self.configuration['clean_stops']:
            result = True if token not in self.configuration['stopwords'] else False
        if self.configuration['clean_punctuation']:
            result = True if token not in self.configuration['punctuation'] else False

        return token, result

    def collect_tokens(self) -> tuple:
        """
        Method is used to collect tokens and labels in the corresponding list
        :return: tuple object contains list of unique tokens of the corpus and list of labels
        """
        tokens = list()
        labels = list()
        source = open(self.configuration['source_data'], 'r')
        count = 0
        print(f"Vocabulary is created for the scenario: "
              f"Cased: {self.configuration['cased']} "
              f"Without stopwords: {self.configuration['clean_stops']} "
              f"Without punctuations: {self.configuration['clean_punctuation']}")
        ti = tqdm(source, desc="Number of successful elements were added to the vocabulary: 0")

        for each_line in ti:
            raw_data = each_line.split(sep='\t')
            if len(raw_data) == 1:
                continue

            token = raw_data[1]
            token, result = self.check_validity(token)

            if result:
                tokens.append(token)
                labels.append(raw_data[2])
                count += 1
                ti.set_description(f"Number of successful elements were added to the vocabulary: {count}")

        self.save_statistics(tokens, labels)

        return [each_token for each_token in set(tokens)], [each_label for each_label in set(labels)]

    def save_statistics(self, tokens: list, labels: list) -> None:
        """
        Method is utilized for collecting token and label distribution in the dataset
        :param tokens: list of tokens
        :param labels: list of labels
        :return: None
        """
        token_counter = Counter(tokens)
        label_counter = Counter(labels)
        stats = os.path.join(self.configuration['processed_folder'], 'statistics.pickle')
        statistics_dict = {
            'tokens': token_counter,
            'labels': label_counter
        }

        with open(stats, 'wb') as stats_out:
            pickle.dump(statistics_dict, stats_out)

    def create_bis_vocabulary(self) -> tuple:
        """
        Method is used to load and define vocabulary and label dictionary for BIS task
        :return: tuple object which contains vocabulary and label dictionaries
        """
        config_file = os.path.join(self.configuration['dataset_dir'], 'bis_params/bis_config.pickle')
        with open(config_file, 'rb') as config_data:
            configuration = pickle.load(config_data)
        return configuration['vocabulary'], configuration['label_dict']

    def create_vocabulary(self) -> tuple:
        """
        Method is used to create vocabulary for the NER task. In order to prevent incompatibility vocabulary for the
        corresponding scenario is saved in the corresponding folder. That is why, vocabulary is generated only once per
        scenario
        :return: tuple that contains vocabulary and label dictionaries
        """
        vocab_file = os.path.join(self.configuration['processed_folder'], 'vocabulary.pickle')
        labels_file = os.path.join(self.configuration['processed_folder'], 'labels_dict.pickle')

        if not os.path.exists(vocab_file):
            unique_tokens, unique_labels = self.collect_tokens()

            vocabulary = {token: idx for idx, token in enumerate(unique_tokens)}
            vocabulary['<UNK>'] = len(vocabulary)
            vocabulary['<PAD>'] = len(vocabulary)

            labels_dict = {label.replace('\n', ''): idx for idx, label in enumerate(unique_labels)}
            labels_dict['<PAD>'] = len(labels_dict)
            with open(vocab_file, 'wb') as vocab_out:
                pickle.dump(vocabulary, vocab_out)

            with open(labels_file, 'wb') as label_out:
                pickle.dump(labels_dict, label_out)

        with open(vocab_file, 'rb') as vocab_out:
            vocabulary = pickle.load(vocab_out)
        with open(labels_file, 'rb') as label_out:
            labels_dict = pickle.load(label_out)
        return vocabulary, labels_dict

    def reverse_labels(self) -> dict:
        """
        Method is used to collect reverse label dictionary which will be used for decoding process
        :return: dictionary in which keys are indexes and values are labels
        """
        return {idx: token for token, idx in self.label_dict.items()}

    def decode(self, idx: int) -> str:
        """
        Method is used to decode provided label index
        :param idx: corresponding label encoding
        :return: label corresponds to the provided information
        """
        if idx not in self.id2label.keys():
            raise IndexError('Index out of range of the vocabulary')
        return self.id2label[idx]

    @property
    def __len__(self) -> int:
        """
        Method is used as a default function of the class, which returns length of the vocabulary
        :return: integer that specifies number of tokens in the vocabulary
        """
        return len(self.vocabulary)

    def __getitem__(self, token: str) -> int:
        """
        Method is used to get index that corresponds to the provided token. In case token does not exist in the
        vocabulary, it returns index for Out Of Vocabulary words -> <UNK>
        :param token: token that corresponding id is requested
        :return:  demanded id from the vocabulary
        """
        if token in self.vocabulary.keys():
            return self.vocabulary[token]
        else:
            return self.vocabulary['<UNK>']
