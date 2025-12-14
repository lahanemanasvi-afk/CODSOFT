## caption_generator.py
import numpy as np
import pickle
import os
import glob
import sys 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import matplotlib.pyplot as plt
# We need to import the custom Attention layer class to load the model
from model_architecture import BahdanauAttention 
# We need ResNet for feature extraction of new images
from tensorflow.keras.applications.resnet_v2 import ResNet101V2 


# --- Configuration (Global Scope) ---
IMG_SIZE = (299, 299)
MAX_LENGTH = 34 
MODEL_PATH = 'model-weights/' 
TOKENIZER_PATH = 'tokenizer.pkl'

# FINAL CONFIRMED FIX: Using the exact confirmed filename and Raw String path.
# Filename: 35506150_cbdb630f4f.jpg
TEST_IMAGE_PATH = r"E:\Image_Captioning_Model\images\23445819_3a458716c1.jpg"
# ------------------------------------


# --- Utility Functions (Code remains the same) ---

def load_tokenizer(path):
    """Loads the saved tokenizer object."""
    try:
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {path}. Ensure 'tokenizer.pkl' is present.")
        sys.exit(1)


def word_for_id(integer, tokenizer):
    """Maps an integer to a word."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def preprocess_new_image(img_path):
    """Loads and preprocesses a new image."""
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features_single(image_input):
    """Uses the ResNet model to extract feature map from a single image."""
    resnet_model = ResNet101V2(weights='imagenet', include_top=False, pooling=None)
    
    feature_map = resnet_model.predict(image_input, verbose=0)
    
    # Reshape for the Attention Model input: (1, H*W, D)
    feature_map_flat = feature_map.reshape(1, -1, feature_map.shape[-1])
    return feature_map_flat


def generate_caption(model, tokenizer, image_features, max_length):
    """Greedy search for generating a caption given an image features."""
    in_text = '<start>'
    
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        
        yhat = model.predict([image_features, np.array([sequence])], verbose=0)
        yhat_index = np.argmax(yhat)
        word = word_for_id(yhat_index, tokenizer)
        
        if word is None or word == '<end>':
            break
            
        in_text += ' ' + word
            
    # Clean up the caption: Remove <start> and <end> tokens first
    final_caption = in_text.replace('<start>', '').replace('<end>', '').strip()
    
    # FIX: Remove any trailing ' end' that was not completely removed by the previous step
    final_caption = final_caption.split(' end')[0].strip() 
    
    return final_caption

# --- Main Execution (Code remains the same) ---

if __name__ == '__main__':
    
    # --- Check if the image path exists ---
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\nFATAL ERROR: Test image not found at {TEST_IMAGE_PATH}")
        print("कृपया तुमच्या 'images/' फोल्डरमध्ये '35506150_cbdb630f4f.jpg' ही फाईल असल्याची खात्री करा.")
        sys.exit(1)
    # --------------------------------------
    
    # 1. Load Tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    # 2. Load Trained Model (Find the best saved model)
    try:
        # Get all saved weight files
        list_of_files = glob.glob(os.path.join(os.getcwd(), MODEL_PATH, '*.keras'))
        
        # Find the best model file based on the lowest loss 
        best_model_file = min(list_of_files, key=lambda f: float(f.split('-loss')[1].replace('.keras', '')))
        
        print(f"\n--- Loading Best Trained Model ---")
        print(f"Loading weights from: {os.path.basename(best_model_file)}")
        
        # Use custom_objects to load the BahdanauAttention layer
        model = load_model(best_model_file, custom_objects={'BahdanauAttention': BahdanauAttention})
        
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load the model. Ensure 'model_architecture.py' is in the folder and BahdanauAttention is correctly imported.")
        print(f"Details: {e}")
        sys.exit(1)

    # 3. Process and Extract Features from New Image
    print(f"\nProcessing Image: {os.path.basename(TEST_IMAGE_PATH)}")
    
    image_input = preprocess_new_image(TEST_IMAGE_PATH)
    
    print("Extracting features (This might take a moment the first time)...")
    image_features = extract_features_single(image_input)

    # 4. Generate Caption
    print("\n--- Generating Caption ---")
    predicted_caption = generate_caption(model, tokenizer, image_features, MAX_LENGTH)

    # 5. Display Result
    print(f"\nPredicted Caption: {predicted_caption}")
    
    # Display the image alongside the caption
    img = load_img(TEST_IMAGE_PATH)
    plt.imshow(img)
    plt.title(f"Predicted Caption:\n{predicted_caption}")
    plt.axis('off')
    plt.show()