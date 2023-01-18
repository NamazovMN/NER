import os
import pickle

from tqdm import tqdm


class ReadDataset:
    """
    Class is an object to read and process dataset
    """

    def __init__(self, config_params: dict, data_type, length_check: bool = False):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_params: configuration parameters include all relevant data for the process
        :param data_type: specifies dataset type that will be collected and processed
        :param length_check: boolean variable that specifies for what class is defined
        """
        self.configuration = self.set_configuration(config_params, data_type, length_check)
        self.dataset = self.read_data()

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

    def set_configuration(self, parameters: dict, dataset_type: str, length_check: bool = False) -> dict:
        """
        Method is used to extract required parameters and return specific parameters dictionary
        :param length_check: boolean variable specifies for what class is called. If true, class skips data process
        :param parameters: configuration parameters dictionary with all parameters for the task
        :param dataset_type: dataset type which can either be training set, test set or dev set
        :return: dictionary with required parameters
        """
        dataset_dir = os.path.join(parameters['dataset_dir'], 'data')
        processed_folder = 'processed'
        if parameters['clean_stops']:
            processed_folder += '_s'
        if parameters['clean_punctuation']:
            processed_folder += '_p'
        if parameters['cased']:
            processed_folder += '_cased'
        processed_folder = self.check_dir(os.path.join(parameters['dataset_dir'], processed_folder))

        window_size, window_shift = self.set_window_size(parameters, length_check)

        configuration = {

            'dataset_path': dataset_dir,
            'processed_path': processed_folder,
            'clean_stops': parameters['clean_stops'],
            'clean_punctuation': parameters['clean_punctuation'],
            'stopwords': parameters['stopwords'],
            'punctuation': parameters['punctuation'],
            'dataset_type': dataset_type,
            'cased': parameters['cased'],
            'vocabulary_obj': parameters['vocabulary'],
            'label_dict': parameters['vocabulary'].label_dict,
            'window_shift': window_shift,
            'window_size': window_size,
            'set_length': parameters['set_length'],
            'set_max_length': parameters['set_max_length'],
        }

        return configuration

    @staticmethod
    def set_window_size(parameters: dict, length_check: bool) -> tuple:
        """
        Method is used to set window size according to set_length and max_length parameters, which are defined by user.
        If set_length is true it will use user's choice for window size and window shift, otherwise the length of the
        longest sequence in the training dataset will be assigned
        :param parameters: configuration parameters dictionary with all parameters for the task
        :param length_check: length_check: boolean variable that specifies for what class is defined
        :return: tuple which contains window size and window shift information
        """
        if length_check or not parameters['set_length']:
            return parameters['window_size'], parameters['window_shift']
        else:
            window_size = parameters['window_size'] if parameters['set_length'] else parameters['max_length']
            window_shift = parameters['window_shift'] if parameters['set_length'] else parameters['max_length']
            return window_size, window_shift

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

    def read_data(self) -> dict:
        """
        Method is used to collect raw data and process them into required form
        :return: dictionary that contains specific dataset
        """
        file_name = os.path.join(self.configuration['processed_path'], f"{self.configuration['dataset_type']}.pickle")
        path = os.path.join(self.configuration['dataset_path'], f"{self.configuration['dataset_type']}.tsv")
        file = open(path, 'r')
        dataset = {
            'data': list(),
            'label': list(),
            'length': list()
        }
        sentence = list()
        annotation = list()

        if not os.path.exists(file_name):
            for each_line in tqdm(file):
                raw_data = each_line.split(sep='\t')
                if len(raw_data) == 1:
                    if len(sentence):
                        dataset['data'].append(sentence)
                        dataset['label'].append(annotation)
                        dataset['length'].append(len(sentence))
                        sentence = list()
                        annotation = list()
                    continue
                token = raw_data[1]
                token, result = self.check_validity(token)
                if result:
                    sentence.append(token)  # token
                    annotation.append(raw_data[2].replace('\n', ''))  # NE

            with open(file_name, 'wb') as out_path:
                pickle.dump(dataset, out_path)
        with open(file_name, 'rb') as out_path:
            dataset = pickle.load(out_path)
        return dataset

    def create_windows(self, sentence, is_label=False):

        windows = list()
        for idx in range(0, len(sentence), self.configuration['window_shift']):
            window = sentence[idx: idx + self.configuration['window_size']]
            difference = self.configuration['window_size'] - len(window)
            if difference:
                window += ['<PAD>'] * difference
            windows.append(self.encode_sentence(window, is_label))
        return windows

    def check_token(self, token, is_label=False):
        """
        Method check whether token exists in vocabulary and indexes it accordingly. If label is provided, checking
        process is skipped
        :param token: can either be character or label
        :param is_label: boolean variable that allows to check existence for labels
        :return: index of token/label in vocabulary/label to idx dictionary
        """
        if is_label:
            return self.configuration['label_dict'][token]
        else:
            return self.configuration['vocabulary_obj'][token]

    def encode_sentence(self, sentence: list, is_label: bool = False) -> list:
        """
        Method encodes provided sentence according to information that sentence is sequence of tokens or labels
        :param sentence: list of tokens/labels in provided sentence
        :param is_label: boolean variable specifies whether provided list is sequence of labels or tokens
        :return: list of encoded tokens/labels according to the provided is_label information
        """
        return [self.check_token(token, is_label) for token in sentence]

    def __iter__(self):
        """
        Method is a default iterator, which helps to iterate over the class object
        :yield: tuple which contains windows for sentence and corresponding label sequence
        """
        for each_sentence, each_label in zip(self.dataset['data'], self.dataset['label']):
            yield self.create_windows(each_sentence), self.create_windows(each_label, is_label=True)

    def __len__(self) -> int:
        """
        Method is used to compute the length of the raw dataset (how many sentences are in the dataset)
        :return: integer specifies how big the dataset is
        """
        return len(self.dataset['label'])

    def __getitem__(self, idx: int) -> dict:
        """
        Method is default getter, which returns dictionary of sequence of tokens and labels
        :param idx: index as a request
        :return: dictionary that contains:
            sentence: sequence of tokens
            label: sequence of labels
        """
        return {
            'sentence': self.dataset['data'][idx],
            'label': self.dataset['label'][idx]
        }
