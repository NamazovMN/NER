import os
import pickle
import torch
from model import Classifier
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score


class Train:
    """
    Class is used to set the environment for training and developing the model
    """

    def __init__(self, config_params: dict, model: Classifier):
        """
        Method initializes the class to set the configuration and initial variables
        :param config_params: configuration parameters include all relevant data for the process
        :param model: Classifier model which will be used for the task
        """
        self.model = model

        self.optimizer = self.set_optimizer(config_params)
        self.configuration = self.set_configuration(config_params)
        self.loss = self.set_loss_fn()

    def set_loss_fn(self) -> nn.CrossEntropyLoss:
        """
        Method is used to set loss function according to the provided parameters
        :return: Loss function
        """
        return nn.CrossEntropyLoss(ignore_index=self.configuration['label_dict']['<PAD>'])

    def set_optimizer(self, parameters: dict) -> Adam:
        """
        Method is used to set optimizer according to the provided parameters
        :param parameters: configuration parameters include all relevant data for the process
        :return: Optimizer for the model
        """
        if parameters['optimizer'] == 'Adam':
            optimizer = Adam(params=self.model.parameters(), lr=parameters['learning_rate'])
        elif parameters['optimizer'] == 'SGD':
            optimizer = SGD(params=self.model.parameters(), lr=parameters['learning_rate'], momentum=0.8)
        else:
            raise Exception('There is not such optimizer in our scenarios. You should choose one of SGD or Adam')
        return optimizer

    @staticmethod
    def check_dir(directory: str) -> str:
        """
        Method checks whether the provided path exist or not. In case it does not exist, path will be created
        :param directory: path, which existence will be checked
        :return: path, which existence was assured
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    @staticmethod
    def save_parameters(output_dir: str, parameters_dict: dict) -> None:
        """
        Method is used to save configuration dictionary to the given path
        :param output_dir: output path where model configuration parameters will be saved
        :param parameters_dict: configuration parameters include all relevant data for the process
        :return: None
        """
        file_name = os.path.join(output_dir, 'model_config.pickle')
        with open(file_name, 'wb') as out_dir:
            pickle.dump(parameters_dict, out_dir)

    def set_configuration(self, parameters):
        """
        Method sets main configuration parameters for this class out of all parameters
        :param parameters: configuration parameters include all relevant data for the process
        :return: dictionary which includes following data:
            'environment': path to raw dataset,
            'out_dir': path to main folder where training and evaluation results will be saved
            'ckpt_dir': path where model checkpoints will be saved
            'device': device on which model will be run
            'num_epochs': number of epochs to train the model
            'label_dict': dictionary for label to idx information
            'exp_num': experiment number specifies specific experiment details to keep
            'resume_training': boolean variable specifies whether resuming the training from the latest epoch is
                               available or not
            'init_eval': boolean variable specifies whether initial evaluation will be done before training or not
            'inference_dir': path to the inference folder where inference results will be saved
        """
        output_dir = self.check_dir('train_results')
        experiment_environment = self.check_dir(os.path.join(output_dir,
                                                             f'experiment_{parameters["experiment_number"]}'))

        self.save_parameters(experiment_environment, parameters)
        checkpoints_directory = self.check_dir(os.path.join(experiment_environment, 'checkpoints'))
        inference_directory = self.check_dir(os.path.join(experiment_environment, 'inferences'))
        return {
            'environment': experiment_environment,
            'out_dir': output_dir,
            'ckpt_dir': checkpoints_directory,
            'device': parameters['device'],
            'num_epochs': parameters['epochs'],
            'label_dict': parameters['vocabulary'].label_dict,
            'exp_num': parameters['experiment_number'],
            'resume_training': parameters['resume_training'],
            'init_eval': parameters['init_eval'],
            'inference_dir': inference_directory
        }

    def compute_acc(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Method is used to compute accuracy according to the provided ground truth and prediction values
        :param prediction: output of the model
        :param target: ground truth labels
        :return: tuple which contains following information:
                correct: number of correct predictions in the provided batch
                length_data: number of data were used to compute the accuracy
                original_predictions: list of predictions, in which tokens from <PAD> positions are discarded
                original_targets: list of targets, in which <PAD> tokens are discarded
        """
        pred_list = torch.argmax(prediction, dim=1).tolist()

        target_list = target.view(-1).tolist()

        original_predictions = list()
        original_targets = list()
        accuracy_list = list()

        for pred, label in zip(pred_list, target_list):
            if label != self.configuration['label_dict']['<PAD>']:
                original_predictions.append(pred)
                original_targets.append(label)
                accuracy_list.append(pred == label)

        correct = sum(accuracy_list)
        length_data = len(original_targets)
        return correct, length_data, original_predictions, original_targets

    def train_step(self, batch_data):
        """
        Method is used to perform training for one step (or for one batch)
        :param batch_data: dictionary which includes train data tensor and label tensor
        :return: tuple which contains following information:
                loss.item(): training loss for this specific batch
                accuracy: accuracy value for this very batch training
                num_tokens: number of tokens that batch includes
        """
        self.optimizer.zero_grad()
        output = self.model(batch_data['data'].to(self.configuration['device']))
        output = output.view(-1, output.shape[-1])
        labels = batch_data['label'].view(-1)

        loss = self.loss(output, labels.to(self.configuration['device']))

        loss.backward()
        self.optimizer.step()
        accuracy, num_tokens, _, _ = self.compute_acc(output, batch_data['label'])
        return loss.item(), accuracy, num_tokens

    def compute_baseline_f1(self, max_label: str, dev_loader: DataLoader) -> float:
        """
        Method is used to compute baseline F1 score for comparison with respect to the label with the highest occurrence
        :param max_label: label which occurred more than others
        :param dev_loader: dataloader for dev data
        :return: f1 score of assumed baseline
        """
        targets = list()
        predictions = list()
        for batch in dev_loader:
            target = batch['label'].view(-1).tolist()
            targets.extend(target)
            predictions.extend([self.configuration['label_dict'][max_label]] * len(target))
        return f1_score(targets, predictions, average='macro')

    def train_epoch(self, data_loader: dict, max_label: str) -> None:
        """
        Method is used to perform training for epochs
        :param data_loader: dictionary contains train and dev dataloaders
        :param max_label: label which occurred more than others
        :return: None
        """
        last_epoch = 0
        baseline_f1 = self.compute_baseline_f1(max_label, data_loader['dev'])
        print(f'Baseline F1 score is computed according to high frequent label {max_label}: {baseline_f1: .3f}')
        if self.configuration['resume_training']:
            print(
                'Please, be sure that model satisfies saved parameters in model_parameters.config in experiment folder')
            last_epoch = self.load_model()
        train_range = range(last_epoch + 1 if not last_epoch == -1 else 0, self.configuration['num_epochs'])
        if self.configuration['init_eval']:
            self.evaluate_epoch(data_loader['dev'], epoch=last_epoch)

        for epoch in train_range:
            print(f'{20 * "<<"} EPOCH {epoch} {20 * ">>"}')
            epoch_loss = 0
            accuracy = 0
            num_batches = len(data_loader['train'])

            ti = tqdm(iterable=data_loader['train'], total=num_batches, leave=True)
            total_tokens = 0
            epoch_accuracy = 0
            for batch in ti:
                self.model.train()
                step_loss, step_accuracy, num_tokens = self.train_step(batch)

                total_tokens += num_tokens
                epoch_accuracy += step_accuracy
                epoch_loss += step_loss
                accuracy += step_accuracy

                ti.set_description(f'Epoch: {epoch}, TRAIN -> epoch loss: {epoch_loss / num_batches : .4f}, '
                                   f'accuracy : {epoch_accuracy / total_tokens : .4f}')

            dev_loss, dev_accuracy, num_batches_dev, total_tokens_dev, f1_dev = self.evaluate_epoch(data_loader['dev'],
                                                                                                    epoch)
            epoch_dict = {
                'train_loss': epoch_loss / num_batches,
                'train_accuracy': accuracy / total_tokens,
                'dev_loss': dev_loss / num_batches_dev,
                'dev_accuracy': dev_accuracy / total_tokens_dev,
                'f1_dev': f1_dev,
                'f1_baseline': baseline_f1
            }
            print(f'F1 score for evaluation data: {f1_dev}')

            self.save_results(epoch_dict, epoch)

    def evaluate_epoch(self, dev_loader: DataLoader, epoch: int) -> tuple:
        """
        Method is used to evaluate the model on dev data after each epoch
        :param dev_loader: DataLoader object for dev data
        :param epoch: current epoch number
        :return: tuple that contains following information:
                dev_loss: validation loss for this epoch
                dev_accuracy: validation accuracy for this epoch
                num_batches: number of batches in validation set
                total_tokens: number of tokens in validation loader
                f1: f1 score for current epoch, that is computed on dev dataset
        """
        num_batches = len(dev_loader)
        dev_loss = 0
        dev_accuracy = 0
        total_tokens = 0
        ti = tqdm(iterable=dev_loader, total=num_batches,
                  desc=f'Epoch: {epoch}, VALIDATION -> epoch loss: {dev_loss}, accuracy : {dev_accuracy}')

        self.model.eval()
        targets = list()
        predictions = list()
        for batch_data in ti:
            output = self.model(batch_data['data'].to(self.configuration['device']))
            output = output.view(-1, output.shape[-1])
            labels = batch_data['label'].view(-1)
            loss = self.loss(output, labels.to(self.configuration['device']))
            acc_dev_step, num_tokens, prediction_sentences, target_sentences = self.compute_acc(output,
                                                                                                batch_data['label'])
            dev_accuracy += acc_dev_step
            dev_loss += loss.item()
            total_tokens += num_tokens
            ti.set_description(f'Epoch: {epoch}, VALIDATION -> epoch loss: {dev_loss / num_batches: .4f}, '
                               f'accuracy : {dev_accuracy / total_tokens :.4f}')
            targets.extend(target_sentences)
            predictions.extend(prediction_sentences)

        f1 = f1_score(targets, predictions, average='macro')
        file_name = os.path.join(self.configuration['inference_dir'], f'inference_epoch_{epoch}.pickle')
        inference_dict = {
            'targets': targets,
            'predictions': predictions
        }
        with open(file_name, 'wb') as inf_data:
            pickle.dump(inference_dict, inf_data)

        return dev_loss, dev_accuracy, num_batches, total_tokens, f1

    def save_results(self, epoch_dict: dict, epoch: int) -> None:
        """
        Method is used to save epoch results according to dynamically provided information after each epoch
        :param epoch_dict: dictionary that includes loss, accuracy and f1 score values of all epoch till the current
        :param epoch: the recent epoch number
        :return: None
        """
        results_dict_file = os.path.join(self.configuration['environment'],
                                         f'lstm_{self.configuration["exp_num"]}_results.pickle')
        if not os.path.exists(results_dict_file):
            result = {
                epoch: epoch_dict
            }
        else:
            with open(results_dict_file, 'rb') as results_dict:
                result = pickle.load(results_dict)
            result[epoch] = epoch_dict

        with open(results_dict_file, 'wb') as result_dict:
            pickle.dump(result, result_dict)

        model_dict_name = os.path.join(self.configuration['ckpt_dir'],
                                       f"epoch_{epoch}_lstm_dev_loss{epoch_dict['dev_loss']: .3f}"
                                       f"_train_loss_{epoch_dict['train_loss']: .3f}"
                                       f"_dev_accuracy_{epoch_dict['dev_accuracy']: .3f}"
                                       f"_f1_{epoch_dict['f1_dev']: .3f}")
        optimizer_dict_name = os.path.join(self.configuration['ckpt_dir'], f'optim_epoch_{epoch}')
        torch.save(self.model.state_dict(), model_dict_name)
        torch.save(self.optimizer.state_dict(), optimizer_dict_name)
        print(f'Model and optimizer parameters for epoch {epoch} were saved successfully!')

    def load_model(self) -> int:
        """
        Method is used to load model for evaluation/ resume training according to the latest checkpoint that was saved
        :return: chosen epoch number
        """
        chosen_epoch = -1
        if not len(os.listdir(self.configuration['ckpt_dir'])):
            print('There is not any saved model parameters! Thus training starts directly!')
        else:
            epochs = list()
            file_names = dict()
            self.configuration['init_eval'] = True
            for each_file_name in os.listdir(self.configuration['ckpt_dir']):
                if 'optim' not in each_file_name:
                    delete_till = each_file_name.index('_lstm')
                    new_name = each_file_name.replace(each_file_name[delete_till::], '')
                    new_name = new_name.replace('epoch_', '')
                    epoch_num = int(new_name)
                    epochs.append(epoch_num)
                    file_names[epoch_num] = each_file_name
            chosen_epoch = max(epochs)
            last_epoch = os.path.join(self.configuration['ckpt_dir'], file_names[chosen_epoch])
            last_optim = os.path.join(self.configuration['ckpt_dir'], f'optim_epoch_{chosen_epoch}')
            self.model.load_state_dict(torch.load(last_epoch, map_location=self.configuration['device']))
            self.optimizer.load_state_dict(torch.load(last_optim, map_location=self.configuration['device']))
            self.model.eval()
            print(f'Model and Optimizer parameters were loaded from checkpoint of epoch {chosen_epoch}, successfully')
        return chosen_epoch
