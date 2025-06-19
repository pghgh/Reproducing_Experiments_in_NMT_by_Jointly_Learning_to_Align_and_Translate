"""
TAKEN FROM 1

The vast majority of the code for performing the experiments with RNNsearch was taken from the following source:
https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

# TAKEN FROM START 1
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

# for debugging purposes, the following abbreviations were used: B - batch, S - sequence, F - features

class Encoder(nn.Module):
    def __init__(self, vocab_length, emb_dim=620, hidden_size=1000, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_length, emb_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden  # output - [B,S,2F], hidden - [2,B,F]


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size=1000):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)  # R^{nxn}
        self.Ua = nn.Linear(2 * hidden_size, hidden_size)  # R^{nx2n}
        self.Va = nn.Linear(hidden_size, 1)  # R^{n}

    def forward(self, query, keys):  # query - [B,1,F], keys - [B,S,2F]
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # [B,S,1]
        scores = scores.squeeze(2)

        weights = F.softmax(scores, dim=-1).unsqueeze(1)  # [B,1,S]
        context = torch.bmm(weights, keys)

        return context, weights  # context - [B,1,2F], weights - [B,1,S]


class Decoder(nn.Module):
    def __init__(self, vocab_length, MAX_LENGTH_SENTENCE, emb_dim=620, hidden_size=1000, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_length, emb_dim)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(emb_dim + 2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_length)
        self.proj = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.MAX_LENGTH_SENTENCE = MAX_LENGTH_SENTENCE

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_token)
        hidden_cat = torch.cat([encoder_hidden[0], encoder_hidden[1]], dim=1)  # [B,2F]
        hidden_projection = torch.tanh(self.proj(hidden_cat)).unsqueeze(0)  # [1,B,F]
        decoder_hidden = hidden_projection
        decoder_outputs = []
        attentions = []

        for i in range(self.MAX_LENGTH_SENTENCE):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)  # [B,1,F]
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

    # TAKEN FROM END 1

    def beam_search(self, encoder_outputs, encoder_hidden, beam_width=3):
        hidden_cat = torch.cat([encoder_hidden[0], encoder_hidden[1]], dim=1)  # [1,1,2F]
        hidden_projection = torch.tanh(self.proj(hidden_cat)).unsqueeze(0)  # [1,F]
        decoder_hidden = hidden_projection

        beams = [(torch.tensor([[SOS_token]]), decoder_hidden, 0.0)]  # [sequence, hidden, score]
        completed_beams = []

        for _ in range(self.MAX_LENGTH_SENTENCE):
            new_beams = []

            for seq, hidden, score in beams:
                decoder_input = seq[:, -1].unsqueeze(1)
                decoder_output, decoder_hidden, _ = self.forward_step(decoder_input, hidden, encoder_outputs)
                probs = F.log_softmax(decoder_output.squeeze(1), dim=-1)
                top_k_probs, top_k_tokens = probs.topk(beam_width)

                for prob, token in zip(top_k_probs.squeeze(), top_k_tokens.squeeze()):
                    new_seq = torch.cat([seq, token.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + prob.item()
                    new_beams.append((new_seq, decoder_hidden, new_score))

            new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]  # keep sorted
            beams = []

            for seq, hidden, score in new_beams:
                if seq[0, -1].item() == EOS_token:
                    completed_beams.append((seq, score))
                else:
                    beams.append((seq, hidden, score))
            if len(beams) == 0: break

        if len(completed_beams) == 0:
            completed_beams = [[seq, score] for seq, _, score in beams]

        best_seq = sorted(completed_beams, key=lambda x: x[1], reverse=True)[0]
        return best_seq[0].squeeze().tolist()
