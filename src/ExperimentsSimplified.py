import sys
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.RNNSearchSimplified import Encoder, Decoder
import torch
import torch.nn as nn
from torch import optim


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=10 ** -6, print_every=100):
    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # L_{2}-norm, according to the paper
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print("print_loss_avg after further 100 epochs", print_loss_avg)
            print_loss_total = 0


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        # teacher forcing is applied by using the original target_tensor
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(decoder_outputs, target_tensor)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    no_epochs = 5
    batch_size = 1
    hidden_dim = 2
    dropout = 0.1

    # start of sentence token
    SOS_token = 0
    # end of sentence token
    EOS_token = 1

    # dummy test data with input and target sentence word ids
    input_ids = [[SOS_token, 2, EOS_token], [SOS_token, 4, EOS_token]]
    target_ids = [[SOS_token, 3, EOS_token], [SOS_token, 5, EOS_token]]

    # in our experiments we won't use sentences which are longer than maximum_sentence_length_allowed
    maximum_sentence_length_allowed = 10
    # -2 is from subtracting SOS_token and EOS_token
    if len(max(input_ids, key=len)) > maximum_sentence_length_allowed:
        sys.exit()  # placeholder

    # https://stackoverflow.com/a/53406082
    # find the longest sentence length from input and target sentences
    longest_sentence_input_len = len(max(input_ids, key=len))
    longest_sentence_target_len = len(max(target_ids, key=len))
    # if needed, make every sentence the same length by adding multiple EOS tokens at the end of the sentences id lists
    if longest_sentence_input_len != longest_sentence_target_len:
        longest_sentence_len = max(len(max(input_ids, key=len)), len(max(target_ids, key=len)))
        ids_list = [input_ids, target_ids]
        for list_with_ids in ids_list:
            for sentence_ids_list in list_with_ids:
                if len(sentence_ids_list) < longest_sentence_len:
                    difference = longest_sentence_len - len(sentence_ids_list)
                    while difference > 0:
                        sentence_ids_list.append(EOS_token)
                        difference -= 1

    # + 2 for the SOS and EOS tokens
    vocab_length_input = len(max(input_ids, key=len))
    vocab_length_target = len(max(input_ids, key=len))
    # embedding dimension should be smaller than any sentence length from input/target in order for the PyTorch Embedding layer to work
    # -2 for the SOS and EOS tokens
    shortest_sentence_len = len(min(input_ids, key=len))
    embedding_dim = len(max(input_ids, key=len)) - 2

    input_ids = torch.tensor(input_ids).to(torch.int64)
    target_ids = torch.tensor(target_ids).to(torch.int64)
    training_data = TensorDataset(input_ids, target_ids)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    encoder = Encoder(vocab_length_input, embedding_dim, hidden_dim, dropout)
    decoder = Decoder(vocab_length_target, embedding_dim, hidden_dim, dropout)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(encoder.to(device))
    print(decoder.to(device))

    train(train_dataloader, encoder, decoder, no_epochs, print_every=1)
    # inference_beam_search
