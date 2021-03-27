import torch
import torch.nn as nn
from tqdm import tqdm

class Training(object):
    def __init__(self, model, optimizer, loss_function, train_dataset, dev_dataset):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

    def train_phase(self, epochs):
        # train_accuracy_list = []
        train_loss_list = []
        # dev_accuracy_list = []
        dev_loss_list = []
        train_loss = 0
        # train_accuracy = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            # total_train = 0
            # correct_train = 0
            for i, data in enumerate(tqdm(self.train_dataset)):
                input_data = data["input_data"].to("cuda")
                labels_data = data["output_data"].to("cuda")
                # print(labels_data)
                self.optimizer.zero_grad()
                outputs = self.model(input_data)
                outputs = outputs.view(-1, outputs.shape[-1])
                labels_data = labels_data.view(-1)
                # print(labels_data)
                sample_loss = self.loss_function(outputs, labels_data)
                sample_loss.backward()

                self.optimizer.step()
                epoch_loss += sample_loss.tolist()

                # _, prediction = torch.max(outputs.data, 1)
                # total_train += labels_data.nelement()
                # correct_train += prediction.eq(labels_data.data).sum().item()

            train_loss_epoch = epoch_loss/len(self.train_dataset)

            # train_accuracy_epoch = correct_train/total_train

            dev_loss_epoch = self.evaluate()

            # dev_accuracy_epoch = self.compute_accuracy()

            train_loss += train_loss_epoch

            # train_accuracy += train_accuracy_epoch

            # train_accuracy_list.append(train_accuracy_epoch)

            train_loss_list.append(train_loss_epoch)

            # dev_accuracy_list.append(dev_accuracy_epoch)

            dev_loss_list.append(dev_loss_epoch)

            print('Epoch:   {}/{}  Train Loss: {:0.4f} Validation Loss:    {:0.4f}'.format(epoch+1, epochs, train_loss_epoch, dev_loss_epoch))
            # print('Epoch:   {}/{}  Train Accuracy: {:0.4f} Validation Accuracy:    {:0.4f}'.format(epoch+1, epochs, train_accuracy_epoch, dev_accuracy_epoch))
        train_loss_avg = train_loss/epochs
        # train_accuracy_avg = train_accuracy/epochs
        
        return train_loss_avg, train_loss_list, dev_loss_list 

    def evaluate(self):
        losses = 0
        self.model.eval()
        with torch.no_grad():
            for i, dataset in enumerate(self.dev_dataset):
                input_data = dataset["input_data"].to("cuda")
                labels_data = dataset["output_data"].to("cuda")
                labels_data = labels_data.view(-1)
                output_data = self.model(input_data)
                output_data = output_data.view(-1, output_data.shape[-1])
                loss = self.loss_function(output_data, labels_data)
                losses += loss.tolist()
        return losses/len(self.dev_dataset)

    def predict(self, input_data):
        """
        Args: 
            input data: Tensor of indices

        output: 
            predictions: list of predicted modelNER tag for each token in the input sentence
            
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_data)
            predictions = torch.argmax(logits, -1)
            return logits, predictions
        
    def compute_accuracy(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            

            for i, dataset in enumerate(self.dev_dataset):
                input_data = dataset["input_data"].to("cuda")
                labels_data = dataset["output_data"].to("cuda")
                labels_data = labels_data.view(-1)

                outputs = self.model(input_data)
                outputs = outputs.view(-1, outputs.shape[-1])

                predicted = torch.argmax(outputs.data, 1)[1]
                # print(predicted.shape)
                total += len(labels_data)
                # print(len(labels_data))
                correct += (predicted == labels_data).sum()

            accuracy = correct / float(total)
            return accuracy