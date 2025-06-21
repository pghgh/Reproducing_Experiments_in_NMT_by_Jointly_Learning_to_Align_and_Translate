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
    criterion = nn.MSELoss()

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
        #decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        # placeholder for decoder_outputs
        decoder_outputs = torch.tensor([[0, 16,  1]]).to(torch.int64)
        decoder_outputs = torch.tensor(decoder_outputs, dtype=torch.float).unsqueeze(0).requires_grad_(True)
        target_tensor = torch.tensor(target_tensor, dtype=torch.float).unsqueeze(0).requires_grad_(True)
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
    dropout = 0.1

    # start of sentence token
    SOS_token = 0
    # end of sentence token
    EOS_token = 1

    # dummy test data with input and target sentence word ids
    # the structure of the list is [[ids for sentence no. 1], ... ,[ids for sentence no. n]]
    # a concrete example: [[SOS_token, id_for_specific_token, ..., id_for_specific_token, EOS_token], ... ,[SOS_token, id_for_specific_token, ..., id_for_specific_token, EOS_token]]
    input_ids = [[SOS_token, 17, EOS_token], [SOS_token, 12, EOS_token]]
    target_ids = [[SOS_token, 13, EOS_token], [SOS_token, 16, EOS_token]]

    # in our experiments we won't use sentences which are longer than maximum_sentence_length_allowed
    maximum_sentence_length_allowed = 10
    if len(max(input_ids, key=len)) > maximum_sentence_length_allowed:
        sys.exit()  # placeholder


    # https://discuss.pytorch.org/t/embedding-error-index-out-of-range-in-self/815
    global_max_element_input = input_ids[0][0]
    for input_ids_list in input_ids:
        if global_max_element_input < max(input_ids_list):
            global_max_element_input = max(input_ids_list)
    global_max_element_target = target_ids[0][0]
    for target_ids_list in target_ids:
        if global_max_element_target < max(target_ids_list):
            global_max_element_target = max(target_ids_list)
    global_max_element = max(global_max_element_input, global_max_element_target)
    vocab_length = global_max_element + 1
    hidden_dim = global_max_element + 2


    input_ids = torch.tensor(input_ids).to(torch.int64)
    target_ids = torch.tensor(target_ids).to(torch.int64)
    training_data = TensorDataset(input_ids, target_ids)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    encoder = Encoder(vocab_length, hidden_dim, dropout)
    decoder = Decoder(vocab_length, hidden_dim, dropout)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(encoder.to(device))
    print(decoder.to(device))

    train(train_dataloader, encoder, decoder, no_epochs, print_every=1)
    # inference_beam_search
