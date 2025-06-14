"""
TAKEN FROM 1

The code for creating the encoder and decoder was taken from the following source:
https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

# TAKEN FROM START 1
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1
MAX_LENGTH_SENTENCE = 10 # maximum length of sentence in words

class EncoderBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size=1000, dropout_p=0.1):
        super(EncoderBiRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size=1000):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size) # R^{nxn}
        self.Ua = nn.Linear(hidden_size, hidden_size) # R^{nx2n}
        self.Va = nn.Linear(hidden_size, 1) # R^{n}

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class DecoderAttentionRNN(nn.Module):
    def __init__(self, output_size, hidden_size=1000, dropout_p=0.1):
        super(DecoderAttentionRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.RNN(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH_SENTENCE):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_rnn = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn(input_rnn, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

# TAKEN FROM END 1

# TODO: integrate beam search into decoder