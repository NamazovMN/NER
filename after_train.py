import torch
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os


class Computations(object):
    def __init__(self, model, vocabulary, label_vocabulary, path_figure, path_model):
        self.model = model
        self.vocabulary = vocabulary
        self.label_vocabulary = vocabulary
        self.path_figure = path_figure

    def compute_precision(self, dataset):
        all_predictions = list()
        all_labels = list()
        for indexed_elem in dataset:
            indexed_input = indexed_elem["input_data"].cuda()
            indexed_labels = indexed_elem["output_data"].cuda()
            predictions = self.model(indexed_input)
            predictions = torch.argmax(predictions, -1).view(-1)
            labels = indexed_labels.view(-1)
            valid_indices = labels != 0

            valid_predictions = predictions[valid_indices]
            valid_labels = labels[valid_indices]

            all_predictions.extend(valid_predictions.tolist())
            all_labels.extend(valid_labels.tolist())
        # global precision. Does take class imbalance into account.
        micro_precision = sk_precision(all_labels, all_predictions, average="micro", zero_division=0)
        # precision per class and arithmetic average of them. Does not take into account class imbalance.
        macro_precision = sk_precision(all_labels, all_predictions, average="macro", zero_division=0)
        per_class_precision = sk_precision(all_labels, all_predictions, labels=list(range(len(self.label_vocabulary))),
                                           average=None, zero_division=0)

        precisions = {"micro_precision": micro_precision, "macro_precision": macro_precision,
                      "per_class_precision": per_class_precision}
        return precisions, all_predictions, all_labels

    def compute_f1_score(self, all_predictions, all_labels):
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_micro = f1_score(all_labels, all_predictions, average='micro')
        return f1_macro, f1_micro

    def compute_confusion_matrix(self, all_predictions, all_labels, model_name):
        path_conf_mat = os.path.join(self.path_figure, model_name + "_conf")
        cf_matrix = confusion_matrix(all_labels, all_predictions)
        sns.heatmap(cf_matrix, annot=True, cmap="coolwarm")
        # sns.color_palette("flare")
        plot0 = plt.figure(1)
        plt.title(model_name)
        plt.savefig(path_conf_mat, format='png')
        plt.close(fig=None)

    def generate_graphs(self, train_list, dev_list, epochs, model_name):
        path_fig = os.path.join(self.path_figure, model_name)
        x_axis = list(range(1, epochs + 1))
        y_axis_train = train_list
        y_axis_dev = dev_list
        plot1 = plt.figure(2)
        plt.plot(x_axis, y_axis_dev)
        plt.plot(x_axis, y_axis_train)
        # plt.show()
        plt.legend(["Validation Loss", "Train Loss"])
        plt.title(model_name)
        plt.savefig(path_fig, format="png")
        plt.close(fig=None)
