import torch
from RNNsearch import Encoder
from experiments import Decoder, SOS_token, EOS_token # Use the fixed Decoder from experiments
from data_loader import normalizeString
import torch.nn.functional as F

def translate_sentence(sentence, encoder, decoder, french_vocab, english_vocab, device, max_length=20):
    """
    Translates a single sentence from French to English.
    """
    # Normalize and tokenize the input sentence
    normalized_sentence = normalizeString(sentence, is_french=True)
    
    # Manually create tensor from sentence without data_loader dependency
    indexes = [french_vocab.word2index.get(word, french_vocab.word2index['UNK']) for word in normalized_sentence.split(' ')]
    input_tensor = torch.tensor([indexes], dtype=torch.long, device=device)

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        # Use beam search for translation
        result_indices = decoder.beam_search(encoder_outputs, encoder_hidden)
        
        # Truncate at EOS
        try:
            eos_index = result_indices.index(EOS_token)
            result_indices = result_indices[:eos_index]
        except (ValueError, AttributeError):
            pass  # No EOS token found

        # Convert indices back to words, excluding SOS
        predicted_words = [english_vocab.index2word.get(idx, 'UNK') for idx in result_indices if idx != SOS_token]
        
    return " ".join(predicted_words)

def main():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # Model parameters (must match the saved model from experiments.py)
    hidden_size = 512
    emb_dim = 256
    max_length = 15
    
    encoder_path = 'best_encoder.pt'
    decoder_path = 'best_decoder.pt'
    french_vocab_path = 'french_vocab.pkl'
    english_vocab_path = 'english_vocab.pkl'
    
    # --- Load Vocabularies and Models ---
    print("Loading vocabularies and models...")
    try:
        french_vocab = torch.load(french_vocab_path)
        english_vocab = torch.load(english_vocab_path)

        # Initialize models
        encoder = Encoder(french_vocab.n_words, emb_dim, hidden_size).to(device)
        decoder = Decoder(english_vocab.n_words, max_length, emb_dim, hidden_size).to(device)

        # Load the saved states
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        
        print("Models and vocabularies loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Make sure you have trained a model and that '{e.filename}' exists.")
        return

    # --- Interactive Translation Loop ---
    print("\nFrench to English Translator")
    print("Enter a French sentence to translate, or type 'quit' to exit.")
    
    while True:
        try:
            input_sentence = input("> ")
            if input_sentence.lower() == 'quit':
                break
            
            translation = translate_sentence(input_sentence, encoder, decoder, french_vocab, english_vocab, device)
            print(f"  -> {translation}")

        except KeyboardInterrupt:
            print("\nExiting translator.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 