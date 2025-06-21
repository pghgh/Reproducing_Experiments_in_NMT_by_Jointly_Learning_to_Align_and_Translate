"""
Data loader for French-English translation dataset
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import re
import unicodedata

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "UNK": 2}
        self.word2count = {"SOS": 0, "EOS": 0, "UNK": 0}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS, EOS, UNK
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s, is_french=False):
    if is_french:
        # For French, preserve accented characters
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÀ-ÿ.!?]+", r" ", s)
    else:
        # For English, use the original normalization
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def readDataFile(filename, max_pairs=20000):
    """Read the French-English dataset"""
    print(f"Reading {filename}...")
    
    # Read the TSV file - it has 4 columns: ID, French, ID, English
    df = pd.read_csv(filename, sep='\t', header=None)
    
    # Clean and normalize the data
    pairs = []
    for i, (_, row) in enumerate(df.iterrows()):
        french = normalizeString(str(row[1]), is_french=True)  # Column 1 is French
        english = normalizeString(str(row[3]))                 # Column 3 is English
        
        # Debug: print first few examples
        if i < 3:
            print(f"Original French: {str(row[1])[:100]}")
            print(f"Normalized French: {french}")
            print(f"Original English: {str(row[3])[:100]}")
            print(f"Normalized English: {english}")
            print("-" * 50)
        
        # Filter out sentences that are too long or too short
        if len(french.split(' ')) < 50 and len(english.split(' ')) < 50:
            if len(french.split(' ')) > 0 and len(english.split(' ')) > 0:
                pairs.append([french, english])
        
        if len(pairs) >= max_pairs:
            break
    
    print(f"Read {len(pairs)} sentence pairs")
    return pairs

def indexesFromSentence(vocab, sentence):
    return [vocab.word2index.get(word, vocab.word2index['UNK']) for word in sentence.split(' ')]

def tensorFromSentence(vocab, sentence, max_length=50, add_eos=True):
    indexes = indexesFromSentence(vocab, sentence)
    if add_eos:
        indexes.append(vocab.word2index['EOS'])
    
    # Pad or truncate to max_length
    if len(indexes) < max_length:
        indexes.extend([vocab.word2index['UNK']] * (max_length - len(indexes)))
    else:
        indexes = indexes[:max_length]
    
    return torch.tensor(indexes, dtype=torch.long)

def tensorFromSentenceForTraining(vocab, sentence, max_length=50):
    """For training - includes EOS token"""
    return tensorFromSentence(vocab, sentence, max_length, add_eos=True)

def tensorFromSentenceForInference(vocab, sentence, max_length=50):
    """For inference - no EOS token"""
    return tensorFromSentence(vocab, sentence, max_length, add_eos=False)

def prepareData(filename, max_pairs=10000, max_length=50):
    """Prepare the dataset for training"""
    pairs = readDataFile(filename, max_pairs)
    
    # Create vocabularies
    french_vocab = Vocabulary('french')
    english_vocab = Vocabulary('english')
    
    for pair in pairs:
        french_vocab.addSentence(pair[0])
        english_vocab.addSentence(pair[1])
    
    print(f"French vocabulary size: {french_vocab.n_words}")
    print(f"English vocabulary size: {english_vocab.n_words}")
    
    # Convert sentences to tensors
    french_tensors = []
    english_tensors = []
    
    for pair in pairs:
        french_tensor = tensorFromSentenceForTraining(french_vocab, pair[0], max_length)
        english_tensor = tensorFromSentenceForTraining(english_vocab, pair[1], max_length)
        french_tensors.append(french_tensor)
        english_tensors.append(english_tensor)
    
    # Create dataset
    french_tensors = torch.stack(french_tensors)
    english_tensors = torch.stack(english_tensors)
    
    dataset = TensorDataset(french_tensors, english_tensors)
    
    return dataset, french_vocab, english_vocab

def createDataLoader(dataset, batch_size=32, shuffle=True):
    """Create a DataLoader from the dataset"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 