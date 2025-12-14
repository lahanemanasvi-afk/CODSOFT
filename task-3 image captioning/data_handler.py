## data_handler.py
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import string
from tqdm import tqdm

def load_clean_descriptions(filename):
    """Loads and cleans captions, adding <start>/<end> tokens."""
    try:
        # We need to manually handle the header line 'image,caption'
        with open(filename, 'r', encoding='utf-8') as f:
            # Read all lines
            lines = f.readlines()
        
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return {}

    descriptions = dict()
    
    # Skip the first line (the header: 'image,caption')
    if lines:
        lines = lines[1:] # This line skips the header row

    # Process the remaining lines (captions)
    for line in lines:
        line = line.strip()
        if not line: # Skip empty lines
            continue
            
        # *** 1. Split using the Comma (,) delimiter ***
        # Using rfind to split only on the last comma, assuming the image name doesn't contain commas
        try:
            split_index = line.index(',')
            image_full_name = line[:split_index].strip()
            caption = line[split_index + 1:].strip()
        except ValueError:
            # Handle lines without a comma if necessary, but typically should be skipped
            continue
        
        if not image_full_name or not caption:
             continue
        
        # *** 2. Extract the base image ID (e.g., '1000268201_693b08cb0e') ***
        # We remove the file extension (.jpg)
        image_id = image_full_name.split('.')[0]
        
        # --- Text Cleaning ---
        caption = caption.lower()
        # Remove punctuation
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        # Remove digits and words shorter than length 1 (basic cleaning)
        # We ensure it only keeps letters
        caption = ' '.join(word for word in caption.split() if len(word) > 1 and word.isalpha())
        
        # Add START and END tokens
        caption = '<start> ' + caption + ' <end>'
        
        # Store the clean caption
        if image_id not in descriptions:
            descriptions[image_id] = list()
        
        descriptions[image_id].append(caption)
        
    return descriptions


def data_generator(descriptions_map, image_features, tokenizer, max_length, vocab_size, batch_size):
    """
    A generator that yields batches of training data in the correct TensorFlow format:
    ((image_features_batch, input_sequences_batch), output_words_onehot_batch).
    """
    X1, X2, y = list(), list(), list()
    n = 0
    
    image_ids = list(descriptions_map.keys())
    
    while True:
        np.random.shuffle(image_ids)
        
        for image_id in image_ids:
            if image_id not in image_features:
                continue

            # Feature map (for attention model)
            feature_map = image_features[image_id]
            # --- FIX FOR NameError: 'captions' is not defined ---
            captions = descriptions_map[image_id]
            # ---------------------------------------------------
            
            for caption in captions:
                # Convert caption to sequence of integer tokens
                seq = tokenizer.texts_to_sequences([caption])[0]
                
                # Create input-output pairs (shifted sequences)
                for i in range(1, len(seq)):
                    # Input Sequence (e.g., <start> a dog)
                    in_seq = seq[:i]
                    # Output Word (e.g., is)
                    out_word = seq[i]
                    
                    # Pad the input sequence
                    in_seq_padded = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    
                    # One-Hot Encode the output word
                    out_word_onehot = to_categorical([out_word], num_classes=vocab_size)[0]
                    
                    X1.append(feature_map)    # Image Feature Map (Encoder Output)
                    X2.append(in_seq_padded)  # Padded Word Sequence (Decoder Input)
                    y.append(out_word_onehot) # Next Word (Decoder Target)
                    
                    n += 1
                    
                    if n == batch_size:
                        # --- FIX FOR TypeError (using Tuple format) ---
                        # Yield the batch in the required Keras/TensorFlow format
                        yield ((np.array(X1), np.array(X2)), np.array(y))
                        X1, X2, y = list(), list(), list()
                        n = 0