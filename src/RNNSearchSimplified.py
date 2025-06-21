import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
# https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://github.com/bentrevett/pytorch-seq2seq/blob/main/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
# https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html#additive-attention
# https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html
# https://www.learnpytorch.io/02_pytorch_classification/

# start of sentence token
SOS_token = 0
# end of sentence token
EOS_token = 1
# in our experiments we won't use sentences which are longer than maximum_sentence_length_allowed
maximum_sentence_length_allowed = 10

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # first apply embedding layer, and then dropout
        embedded_result_with_dropout = self.dropout(self.embedding(input))
        # output = output of last layer, hidden = last hidden layer
        output, last_hidden_layer = self.gru(embedded_result_with_dropout)
        return output, last_hidden_layer

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim)
        #self.attention = Attention()
        self.gru = nn.GRU(input_size=2 * hidden_dim, hidden_size=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(input_dim, hidden_dim)