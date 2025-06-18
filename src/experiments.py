"""
TAKEN FROM 1

The vast majority of the code for creating the encoder and decoder was taken from the following source:
https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

# TAKEN FROM START 1
from torch.utils.data import DataLoader, TensorDataset
from src.RNNsearch import EncoderBiRNN, DecoderAttentionRNN
import torch
import torch.nn as nn
from torch import optim


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


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=10 ** -6, print_every=100):
    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print("print_loss_avg after further 100 epochs", print_loss_avg)
            print_loss_total = 0
# TAKEN FROM END 1

# works with batch=1 only
def inference_beam_search(dataloader, encoder, decoder):
    decoder.eval()
    encoder.eval()
    with torch.no_grad():
        for input_tensor, _ in dataloader:
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            result = decoder.beam_search(encoder_outputs, encoder_hidden)
            print(result)



# TAKEN FROM START 1
if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)

    # architecture predefined params (though definitely should be decreased when testing, these are the numbers from the paper)
    hidden_size = 1000
    emb_dim = 620
    batch_size = 80

    # vocab length for both languages
    vocab_length1 = 10
    vocab_length2 = 10

    no_epochs = 5
    encoder = EncoderBiRNN(vocab_length1, emb_dim, hidden_size)
    decoder = DecoderAttentionRNN(vocab_length2, emb_dim, hidden_size)

    # dummy sentences which were converted as integers (with ids)
    sentences_list = ["i saw a black cat", "j'ai vu un chat noir"]
    integer_to_word_en = {0: "SOS_token", 1: "EOS_token", 2: "i", 3: "saw", 4: "a", 5: "black", 6: "cat"}
    integer_to_word_fr = {0: "SOS_token", 1: "EOS_token", 2: "j", 3: "ai", 4: "vu", 5: "un", 6: "chat", 7: "noir"}
    # the structure of the list is [language_id, [SOS_token, id_1, id_2, ..., id_n, EOS_token]]
    input_ids = [[0, 2, 3, 4, 5, 6, 1]]
    target_ids = [[0, 2, 3, 4, 5, 6, 1]]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    target_ids = torch.tensor(target_ids, dtype=torch.long)
    training_data = TensorDataset(input_ids, target_ids)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    train(train_dataloader, encoder, decoder, no_epochs, print_every=1)
    inference_beam_search(DataLoader(training_data, batch_size=1, shuffle=True), encoder, decoder)

# TAKEN FROM END 1