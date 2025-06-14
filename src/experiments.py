"""
TAKEN FROM 1

The code for creating the encoder and decoder was taken from the following source:
https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

# TAKEN FROM START 1
from torch.utils.data import DataLoader, TensorDataset
from src.RNNsearch import EncoderBiRNN, DecoderAttentionRNN
import torch
import torch.nn as nn
from torch import optim
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        # reset gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=10 ** -6,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # squared L2 norm

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == "__main__":
    seed = 1
    hidden_size = 1000
    batch_size = 80

    encoder = EncoderBiRNN(10, hidden_size)
    decoder = DecoderAttentionRNN(hidden_size, 10)

    # dummy sentences which were converted as integers
    sentences_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    training_data = TensorDataset(torch.tensor([sentences_list]))
    train_dataloader = DataLoader(training_data, batch_size=80, shuffle=True)
    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

# TAKEN FROM END 1

# TODO: modify sentences_list so that the code runs & potentially rewrite evaluation part