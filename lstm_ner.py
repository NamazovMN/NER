import torch
import torch.nn as nn

class LSTM_NER(nn.Module):
    def __init__(self, vocabulary, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob = 0.4, bidirectional = True):
        super(LSTM_NER, self).__init__()

        self.output_size = output_size
        if bidirectional:
            lstm_out = 2*hidden_dim
        else:
            lstm_out = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = vocabulary["<pad>"])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional= bidirectional, num_layers = n_layers, dropout = drop_prob, batch_first = True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(lstm_out,output_size)
        # self.relu = nn.ReLU()

    def forward(self, x):
        embeddings = self.dropout(self.embedding(x))
        out, (hidden, cell_state) = self.lstm(embeddings)
        output = self.dropout(out)
        output = self.fc(output)
        # output = self.relu(output)

        return output

