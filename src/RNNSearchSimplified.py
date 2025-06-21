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
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # first apply embedding layer, and then dropout
        embedded_result_with_dropout = self.dropout(self.embedding(input))
        # output = output of last layer, hidden = last hidden layer
        output, last_hidden_layer = self.gru(embedded_result_with_dropout)
        return output, last_hidden_layer


class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden_layer, target_tensor):
        input_size = input.size(1)
        hidden_layer = hidden_layer.squeeze(0)
        decoder_input = torch.empty(input_size, 1, dtype=torch.long).fill_(SOS_token)
        decoder_outputs = []


        for i in range(maximum_sentence_length_allowed):
            decoder_output, hidden_layer  = self.forward_step(decoder_input)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, hidden_layer, None # We return `None` for consistency in the training loop


    def forward_step(self, input):
        embedded_result_with_dropout = self.dropout(self.embedding(input))
        embedded_result_with_dropout = embedded_result_with_dropout.squeeze(1)
        output, hidden_layer = self.gru(embedded_result_with_dropout)
        return self.attention(output, hidden_layer)



# based on the Bahdanau/Additive Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # weight matrices as defined in the paper
        self.Wa = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)  # R^{nxn}
        self.Ua = nn.Linear(in_features=hidden_dim, out_features=2 * hidden_dim)  # R^{nx2n}
        self.va = nn.Linear(in_features=1, out_features=hidden_dim)  # R^{n}


    def forward(self, query, key):
        key = torch.transpose(key, dim0=0, dim1=1)
        sum = self.Wa(query) + self.Ua(key)
        tanh = torch.tanh(sum)
        scores = self.va(tanh)
        #scores = self.va(torch.tanh(self.Wa(query) + self.Ua(key)))  # [B,S,1]
        weights = F.softmax(scores)
        context = torch.bmm(weights, key)
        return context, weights



