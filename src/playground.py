import os
import pickle
import torch
from model import Classifier
from set_bis import BISClassifier
import nltk
from nltk.tokenize import word_tokenize
from vocab import Vocabulary

nltk.download('punkt')


class Playground:
    """
    This class is built for testing and playing around with the model.
    """

    def __init__(self, config_params: dict, model: Classifier):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_params: configuration parameters include all relevant data for the process
        :param model: Classifier model which will be used for the task
        """
        self.configuration = self.set_configuration(config_params)
        self.bis_configuration = self.load_bis_configuration()
        self.bis_model = BISClassifier(self.bis_configuration).to(self.configuration['device'])
        self.model = model

    def load_bis_configuration(self) -> dict:
        """
        Method is used to collect BIS Token Classifier parameters
        :return: BIS Classifier parameters in form of dictionary
        """
        config_path = os.path.join(self.configuration['dataset_dir'], 'bis_params/bis_config.pickle')
        with open(config_path, 'rb') as config_bis:
            bis_config = pickle.load(config_bis)
        return bis_config

    def load_bis_model(self) -> None:
        """
        Method is used to load BIS Classifier Model. Note: Model Configuration must be same as in model object!
        :return: None
        """
        model_path = os.path.join(self.configuration['dataset_dir'], 'bis_params')

        model_path = os.path.join(model_path,
                                  'epoch_6_lstm_dev_loss 0.333_train_loss_ 0.329_dev_accuracy_ 0.867_f1_ 0.805')
        self.bis_model.load_state_dict(torch.load(model_path, map_location=self.configuration['device']))

        self.bis_model.eval()

    def set_inverse_vocab(self, is_data: bool, is_ner: bool = True) -> dict:
        """
        Method is used to generate idx to vocabulary / label dictionary according to vocabulary / label dictionary
        :param is_data: boolean variable specifies whether tokens (True) or labels (False) will be processed
        :param is_ner: boolean variable specifies whether reverse dictionary is made for NER or BIS
        :return: dictionary which keys are indexes and values are corresponding tokens (True) / labels (False)
        """
        if is_ner:
            main_source = self.configuration['vocabulary'] if is_data else self.configuration['label_dict']
        else:
            main_source = self.bis_configuration['vocabulary'] if is_data else self.bis_configuration['label_dict']
        return {idx: token for token, idx in main_source.items()}

    @staticmethod
    def set_configuration(parameters: dict) -> dict:
        """
        Method sets main configuration parameters for this class out of all parameters
        :param parameters: configuration parameters include all relevant data for the process
        :return: dictionary which includes following data:
                'checkpoints_dir': path to the checkpoints of each epoch
                'environment': main experimental environment, in which all corresponding train results are kept
                'experiment_num': number of the experiment which is significant to eliminate confusion
                'vocabulary': vocabulary dictionary that is used to encode tokens
                'label_dict': dictionary to encode labels
                'cased': specifies whether model is case-sensitive or not
                'device': device that model was trained on
                'window_shift': step size for shifting the window
                'window_size': number of tokens that window can contain
        """

        experiment_environment = f"../train_results/experiment_{parameters['experiment_number']}"
        checkpoints_dir = os.path.join(experiment_environment, 'checkpoints')
        bis_vocabulary = Vocabulary(parameters, is_bis=True)
        return {
            'dataset_dir': parameters['dataset_dir'],
            'checkpoints_dir': checkpoints_dir,
            'environment': experiment_environment,
            'experiment_num': parameters['experiment_number'],
            'vocabulary': parameters['vocabulary'],
            'label_dict': parameters['vocabulary'].label_dict,
            'bis_vocabulary': bis_vocabulary,
            'bis_label_dict': bis_vocabulary.label_dict,
            'cased': parameters['cased'],
            'device': parameters['device'],
            'window_shift': parameters['window_shift'],
            'window_size': parameters['window_size'],
            'punctuation': parameters['punctuation'],
            'play_bis': parameters['play_bis']
        }

    def get_best_model(self) -> tuple:
        """
        Method is used to filter epochs according to the F1 score. Path to the best model is returned
        :return: path to the best model parameters
        """
        result_file = os.path.join(self.configuration['environment'],
                                   f"lstm_{self.configuration['experiment_num']}_results.pickle")
        if os.path.exists(result_file):
            with open(result_file, 'rb') as result_data:
                result_dict = pickle.load(result_data)
        else:
            raise FileNotFoundError('There is not such file. It occurs, because no training happened!')
        specific_dict = {epoch: result['f1_dev'] for epoch, result in result_dict.items()}
        best_epoch = max(specific_dict, key=specific_dict.get)
        best_path = str()
        for ckpt in os.listdir(self.configuration['checkpoints_dir']):
            if f'epoch_{best_epoch}_' in ckpt:
                best_path = ckpt
                break
        return os.path.join(self.configuration['checkpoints_dir'], best_path), best_epoch


    def load_model(self) -> None:
        """
        Model is used to load state dictionary of the best model which was chosen according to F1 score result
        """
        best_model_path, _ = self.get_best_model()
        self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

    def remap_windows(self, windows: list, sentence: list, is_ner: bool = True) -> list:
        """
        Method is used to transform prediction data into given shape, since we modify it according to the training
        structure
        :param windows: list of windows, in which predicted labels are kept in form of windows
        :param sentence: original sentence, which is used to check data reconfiguration
        :param is_ner: boolean variable specifies whether reverse dictionary is made for NER or BIS
        :return: list of labels in the same length of the sentence
        """

        config_dict = self.configuration if is_ner else self.bis_configuration
        pred_windows = list()
        for idx in range(0, len(sentence), config_dict['window_shift']):
            pred_windows.append(sentence[idx: idx + config_dict['window_size']])

        check_slice = slice(0, config_dict['window_shift'], 1)

        req_preds = list()
        req_originals = list()
        for pred, original in zip(pred_windows, windows):
            req_org = original[check_slice]
            req_pred = pred[check_slice]
            if '<PAD>' in original[check_slice]:
                idx = req_org.index('<PAD>')
                req_pred = req_pred[0: idx]
                req_org = req_org[0: idx]

            req_preds.append(req_pred)
            req_originals.append(req_org)

        predictions = list()
        for each in req_preds:
            predictions.extend(each)
        return predictions

    def decode_sentence(self, sentence: list, windows: list, is_ner: bool = True) -> list:
        """
        Method is used to decode sentence according to its 'windowed' version
        :param sentence: list of tokens in the provided sentence
        :param windows: windows of the sentence for encoding
        :param is_ner: boolean variable specifies whether decoding is done for NER (True) or BIS (False)
        :return: decoded (in terms of labels) version of the provided sentence
        """
        vocabulary = self.configuration['vocabulary'] if is_ner else self.configuration['bis_vocabulary']
        pred_sentence = self.remap_windows(windows, sentence, is_ner)
        decoded = [vocabulary.decode(token) for token in pred_sentence]
        return decoded

    def encode_token(self, token: str, is_ner: bool = True) -> str:
        """
        Method is used to encode provided token
        :param token: character from the sentence
        :param is_ner: boolean variable specifies whether encoding is done for NER (True) or BIS (False)
        :return: encoding of the provided token
        """

        config_dict = self.configuration if is_ner else self.bis_configuration
        vocabulary = self.configuration['vocabulary'] if is_ner else self.configuration['bis_vocabulary']
        token = token if config_dict['cased'] else token.lower()
        return vocabulary[token]

    def encode_sentence(self, sentence: list, is_ner: bool = True) -> tuple:
        """
        Method is used to encode provided sentence
        :param sentence: list of tokens of the sentence
        :param is_ner: boolean variable specifies whether sentence encoding is done for NER (True) or BIS (False)
        :return: tuple that contains:
                encoded_window: list of windows in which tokens are kept as encoded
                data: list of windows in which raw tokens are kept
        """
        config_dict = self.configuration if is_ner else self.bis_configuration
        data = list()
        for idx in range(0, len(sentence), config_dict['window_shift']):
            window = sentence[idx: idx + config_dict['window_size']]
            if len(window) < config_dict['window_size']:
                window += ['<PAD>'] * (config_dict['window_size'] - len(window))
            data.append(window)
        encoded_window = list()
        for each_window in data:
            enc_wind = [self.encode_token(token, is_ner) for token in each_window]
            encoded_window.append(enc_wind)

        return encoded_window, data

    def run_inference(self, sentence: list, is_ner: bool = True) -> list:
        """
        Method is used to perform inference with following order:
            - Load the model
            - encode the sentence
            - predict
            - decode prediction
            - print result
        :param sentence: input sentence which is provided by user
        :param is_ner:
        :return: None
        """
        self.load_model() if is_ner else self.load_bis_model()

        encoded_sentence, data_form = self.encode_sentence(sentence, is_ner=is_ner)
        input_data = torch.LongTensor(encoded_sentence).to(self.configuration['device'])
        output = self.model(input_data) if is_ner else self.bis_model(input_data)
        prediction = torch.argmax(output.view(-1, output.shape[-1]), dim=1).tolist()
        result = self.decode_sentence(prediction, data_form, is_ner=is_ner)
        return result

    def tokenize(self, sentence: list, result: list) -> list:
        """
        Method is used to tokenize provided sentence according to BIS Token Classification Model
        :param sentence: input sentence from the user
        :param result: classified tokens in the given sentence, which will be uses for tokenization
        :return: list of tokens in the provided sentence
        """

        word = list()
        tokenized_sentence = list()

        for idx, (letter, label) in enumerate(zip(sentence, result)):
            if label == 'S':
                tokenized_sentence.append(''.join(word))
                word = list()
                continue
            elif letter in self.configuration['punctuation']:
                tokenized_sentence.append(''.join(word))
                tokenized_sentence.append(letter)
                word = list()
                continue
            elif idx == len(sentence) - 1:

                word.append(letter)
                tokenized_sentence.append(''.join(word))
                break

            word.append(letter)

        return [each for each in tokenized_sentence if each != '']

    def process(self):
        """
        Method is used as main function of playground phase. It takes sentence of the user and send it for process
        :return: None
        """
        sentence = input('Please provide your sentence: ')
        chosen_sentence = [token for token in sentence] if self.configuration['play_bis'] else sentence

        result_bis = self.run_inference(chosen_sentence, is_ner=False)
        tokenized_sentence = self.tokenize(chosen_sentence, result_bis) if self.configuration['play_bis'] \
            else word_tokenize(chosen_sentence)

        ner_classified = self.run_inference(tokenized_sentence)
        print('Classification result:')
        print(f'Input: {sentence}')
        print(f'Input Tokenized: {tokenized_sentence}')
        print(f'NER: {ner_classified}')
