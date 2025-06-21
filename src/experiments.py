"""
TAKEN FROM 1

The vast majority of the code for creating the encoder and decoder was taken from the following source:
https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

# TAKEN FROM START 1
from torch.utils.data import DataLoader, TensorDataset
from RNNsearch import Encoder, Decoder as OriginalDecoder, BahdanauAttention
from data_loader import prepareData, createDataLoader
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define token constants locally to avoid modifying RNNsearch.py
SOS_token = 0
EOS_token = 1

# Download required NLTK data for BLEU score
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Create a new, device-aware Decoder class that inherits from the original
class Decoder(OriginalDecoder):
    """
    A device-aware version of the Decoder from RNNsearch.py.
    This version overrides the `forward` and `beam_search` methods to ensure
    that all tensors are created on the correct device (CPU or GPU),
    preventing the common "tensors on different devices" runtime error.
    """
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        device = encoder_outputs.device  # Infer device from an input tensor
        batch_size = encoder_outputs.size(0)
        # Create decoder_input on the correct device
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        
        hidden_cat = torch.cat([encoder_hidden[0], encoder_hidden[1]], dim=1)
        hidden_projection = torch.tanh(self.proj(hidden_cat)).unsqueeze(0)
        decoder_hidden = hidden_projection
        decoder_outputs = []
        attentions = []

        for i in range(self.MAX_LENGTH_SENTENCE):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            # This implementation doesn't use teacher forcing, so the next decoder_input
            # is implicitly handled inside the loop for free generation.
            # We'll just use the last output as the input for the next step.
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def beam_search(self, encoder_outputs, encoder_hidden, beam_width=3):
        device = encoder_outputs.device  # Infer device from an input tensor
        hidden_cat = torch.cat([encoder_hidden[0], encoder_hidden[1]], dim=1)
        hidden_projection = torch.tanh(self.proj(hidden_cat)).unsqueeze(0)
        decoder_hidden = hidden_projection

        # Create initial beams tensor on the correct device
        beams = [(torch.tensor([[SOS_token]], device=device), decoder_hidden, 0.0)]
        completed_beams = []

        for _ in range(self.MAX_LENGTH_SENTENCE):
            new_beams = []
            if not beams:
                break
            
            for seq, hidden, score in beams:
                decoder_input = seq[:, -1].unsqueeze(1)
                decoder_output, new_hidden, _ = self.forward_step(decoder_input, hidden, encoder_outputs)
                probs = F.log_softmax(decoder_output.squeeze(1), dim=-1)
                top_k_probs, top_k_tokens = probs.topk(beam_width)

                for i in range(beam_width):
                    token = top_k_tokens[0][i].unsqueeze(0).unsqueeze(0)
                    prob = top_k_probs[0][i]

                    new_seq = torch.cat([seq, token], dim=1)
                    new_score = score + prob.item()

                    if token.item() == EOS_token:
                        completed_beams.append((new_seq, new_score))
                    else:
                        new_beams.append((new_seq, new_hidden, new_score))
            
            # Prune beams
            completed_beams = sorted(completed_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

        # If no beams completed, take the best of the running beams
        if not completed_beams:
             completed_beams = [(b[0], b[2]) for b in beams]

        if not completed_beams:
            return []

        best_seq, _ = sorted(completed_beams, key=lambda x: x[1], reverse=True)[0]
        return best_seq.squeeze().tolist()


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, device, clip=1.0):
    total_loss = 0
    encoder.train()
    decoder.train()
    for data in dataloader:
        input_tensor, target_tensor = data
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

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

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_epoch(dataloader, encoder, decoder, criterion):
    """Evaluates the model on the validation set for one epoch."""
    total_loss = 0
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, device, learning_rate=0.001, print_every=1):
    train_loss_history = []

    # Use AdamW optimizer for better weight decay handling
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
    
    # Add a learning rate scheduler
    encoder_scheduler = StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    decoder_scheduler = StepLR(decoder_optimizer, step_size=20, gamma=0.5)
    
    criterion = nn.NLLLoss()

    best_train_loss = float('inf')
    print("Starting training...")

    for epoch in range(1, n_epochs + 1):
        # Training step
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        train_loss_history.append(train_loss)
        
        # Update learning rate
        encoder_scheduler.step()
        decoder_scheduler.step()

        if epoch % print_every == 0:
            print(f"Epoch {epoch}/{n_epochs} | Loss: {train_loss:.4f}")

        # Save the best model based on training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            print(f"  -> New best training loss: {best_train_loss:.4f}. Saving model...")
            torch.save(encoder.state_dict(), 'best_encoder.pt')
            torch.save(decoder.state_dict(), 'best_decoder.pt')
    
    print("Finished training.")
    return train_loss_history

def plot_loss_graph(train_loss, save_path='loss_graph.png'):
    """Plot the training loss over epochs"""
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, 'b-o', label='Training Loss', markersize=4)
    plt.title('Training Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Add some statistics
    final_train_loss = train_loss[-1]
    min_train_loss = min(train_loss)
    plt.text(0.02, 0.98, f'Final Loss: {final_train_loss:.4f}\nMin Loss: {min_train_loss:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Loss graph saved to {save_path}")

# TAKEN FROM END 1

# works with batch=1 only
def inference_beam_search(dataloader, encoder, decoder, french_vocab, english_vocab, device, num_examples=5):
    decoder.eval()
    encoder.eval()
    count = 0
    with torch.no_grad():
        for input_tensor, target_tensor in dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            if count >= num_examples:
                break
                
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            result = decoder.beam_search(encoder_outputs, encoder_hidden)
            
            # Truncate the result at the first EOS token
            try:
                eos_index = result.index(EOS_token)
                result = result[:eos_index]
            except (ValueError, AttributeError):
                pass  # No EOS token found or result is not a list
            
            # Convert indices back to words
            input_sentence = ' '.join([french_vocab.index2word.get(idx.item(), 'UNK') for idx in input_tensor[0] if idx.item() != EOS_token and idx.item() != SOS_token])
            target_sentence = ' '.join([english_vocab.index2word.get(idx.item(), 'UNK') for idx in target_tensor[0] if idx.item() != EOS_token and idx.item() != SOS_token])
            predicted_sentence = ' '.join([english_vocab.index2word.get(idx, 'UNK') for idx in result if idx != SOS_token])
            
            print(f"Input (French): {input_sentence}")
            print(f"Target (English): {target_sentence}")
            print(f"Predicted (English): {predicted_sentence}")
            print("-" * 50)
            count += 1

def calculate_bleu_score(dataloader, encoder, decoder, french_vocab, english_vocab, device, num_samples=100):
    """Calculate BLEU score on a dataset"""
    decoder.eval()
    encoder.eval()
    
    references = []
    hypotheses = []
    smoothing = SmoothingFunction().method1
    
    count = 0
    with torch.no_grad():
        for input_tensor, target_tensor in dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            if count >= num_samples:
                break
                
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            result = decoder.beam_search(encoder_outputs, encoder_hidden)
            
            # Truncate the result at the first EOS token
            try:
                eos_index = result.index(EOS_token)
                result = result[:eos_index]
            except (ValueError, AttributeError):
                pass # No EOS token found or result is not a list
            
            # Convert indices back to words
            target_sentence = ' '.join([english_vocab.index2word.get(idx.item(), 'UNK') for idx in target_tensor[0]])
            predicted_sentence = ' '.join([english_vocab.index2word.get(idx, 'UNK') for idx in result])
            
            # Remove SOS, EOS, and UNK tokens for BLEU calculation
            target_words = [word for word in target_sentence.split() if word not in ['SOS', 'EOS', 'UNK']]
            predicted_words = [word for word in predicted_sentence.split() if word not in ['SOS', 'EOS', 'UNK']]
            
            references.append([target_words])  # BLEU expects list of reference sentences
            hypotheses.append(predicted_words)
            
            count += 1
    
    # Calculate BLEU scores
    bleu_1 = 0
    bleu_2 = 0
    bleu_3 = 0
    bleu_4 = 0
    
    for ref, hyp in zip(references, hypotheses):
        if len(hyp) > 0:  # Only calculate if hypothesis is not empty
            bleu_1 += sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 += sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 += sentence_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_4 += sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    num_valid = len([h for h in hypotheses if len(h) > 0])
    if num_valid > 0:
        bleu_1 /= num_valid
        bleu_2 /= num_valid
        bleu_3 /= num_valid
        bleu_4 /= num_valid
    
    return bleu_1, bleu_2, bleu_3, bleu_4

def translate_sentence(sentence, encoder, decoder, french_vocab, english_vocab, device):
    """Translate a single sentence"""
    # Normalize and tokenize the input sentence
    from data_loader import normalizeString, tensorFromSentenceForInference
    
    normalized = normalizeString(sentence, is_french=True)
    input_tensor = tensorFromSentenceForInference(french_vocab, normalized).unsqueeze(0).to(device)
    
    decoder.eval()
    encoder.eval()
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        result = decoder.beam_search(encoder_outputs, encoder_hidden)
        
        # Truncate the result at the first EOS token
        try:
            eos_index = result.index(EOS_token)
            result = result[:eos_index]
        except (ValueError, AttributeError):
            pass # No EOS token found or result is not a list
            
        # Convert indices back to words
        predicted_sentence = ' '.join([english_vocab.index2word.get(idx, 'UNK') for idx in result if idx != SOS_token])
        return predicted_sentence

# TAKEN FROM START 1
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # Data parameters
    data_file = "../Fr-Eng.tsv"
    max_pairs = 10000  # or 50000, 100000
    max_length = 15   # Maximum sentence length
    
    # Model parameters
    hidden_size = 512  # Reduced for faster training
    emb_dim = 256      # Reduced for faster training
    batch_size = 32
    no_epochs = 50   
    learning_rate = 0.001
    print("Preparing French-English dataset...")
    dataset, french_vocab, english_vocab = prepareData(data_file, max_pairs, max_length)

    # Split dataset into training and testing sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test sets
    train_dataloader = createDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = createDataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Total dataset size: {len(dataset)}")
    print(f"  - Training set size: {len(train_dataset)}")
    print(f"  - Testing set size:  {len(test_dataset)}")
    print(f"French vocabulary size: {french_vocab.n_words}")
    print(f"English vocabulary size: {english_vocab.n_words}")

    # Save the vocabularies for use in translation script
    print("Saving vocabularies...")
    torch.save(french_vocab, 'french_vocab.pkl')
    torch.save(english_vocab, 'english_vocab.pkl')

    # Create encoder and our new device-aware decoder
    encoder = Encoder(french_vocab.n_words, emb_dim, hidden_size).to(device)
    decoder = Decoder(english_vocab.n_words, max_length, emb_dim, hidden_size).to(device)

    # Train the model and get loss history
    train_loss_history = train(train_dataloader, encoder, decoder, no_epochs, device, learning_rate, print_every=1)
    
    print("\nLoading best model for inference...")
    # Load the model states onto the correct device
    encoder.load_state_dict(torch.load('best_encoder.pt', map_location=device))
    decoder.load_state_dict(torch.load('best_decoder.pt', map_location=device))
    
    print("\nRunning inference on a few test samples with the best model...")
    inference_beam_search(test_dataloader, encoder, decoder, french_vocab, english_vocab, device, num_examples=5)
    
    # Test with a custom sentence
    test_sentence = "bonjour comment allez vous"
    translation = translate_sentence(test_sentence, encoder, decoder, french_vocab, english_vocab, device)
    print(f"\nCustom translation test:")
    print(f"French: {test_sentence}")
    print(f"English: {translation}")

    # Calculate BLEU scores on the full test set
    print("\nCalculating BLEU score on the full test set...")
    bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_score(test_dataloader, encoder, decoder, french_vocab, english_vocab, device, num_samples=len(test_dataset))
    print(f"\nBLEU-1 score on test set: {bleu_1:.4f}")
    print(f"BLEU-2 score on test set: {bleu_2:.4f}")
    print(f"BLEU-3 score on test set: {bleu_3:.4f}")
    print(f"BLEU-4 score on test set: {bleu_4:.4f}")

    # Plot loss graph
    plot_loss_graph(train_loss_history)

# TAKEN FROM END 1
