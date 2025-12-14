## main_train.py
from model_architecture import create_attention_model 
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
# Add necessary imports for training
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
from data_handler import load_clean_descriptions, data_generator 

# --- Configuration ---
CAPTION_FILE_PATH = 'captions.txt'
FEATURE_FILE_PATH = 'image_features_resnet101v2.npy'
VOCAB_SIZE_LIMIT = 5000 # Limit vocabulary size
BATCH_SIZE = 64
EPOCHS = 20 # You may need 15-30 epochs for good results

# ----------------------------------------------------

## 1. Text Data Loading and Tokenization
print("--- Starting Text Preprocessing ---")

# Load and clean the raw captions
raw_descriptions = load_clean_descriptions(CAPTION_FILE_PATH)
if not raw_descriptions:
    print("ERROR: No descriptions loaded. Check if captions.txt exists and its format is correct.")
    exit()

print(f"Loaded {len(raw_descriptions)} unique image descriptions.")

# Get all captions into a single list
all_captions = [cap for img_id, caps in raw_descriptions.items() for cap in caps]

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE_LIMIT)
tokenizer.fit_on_texts(all_captions)

# Determine final vocabulary size
VOCAB_SIZE = len(tokenizer.word_index) + 1 
print(f"Final Vocabulary Size: {VOCAB_SIZE}")

# Determine max sequence length (needed for padding)
max_length = max(len(cap.split()) for cap in all_captions)
print(f"Max Caption Length: {max_length}")

# Save the tokenizer for inference later
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Text preprocessing complete. Tokenizer saved.")
print("---------------------------------------------")


## 2. Load Image Features (Encoder Output)
print("--- Loading Image Features ---")
if not os.path.exists(FEATURE_FILE_PATH):
    print(f"ERROR: Feature file not found at {FEATURE_FILE_PATH}. Run feature_extractor.py first!")
    exit()
    
# Load the features created by feature_extractor.py
image_features = np.load(FEATURE_FILE_PATH, allow_pickle=True).item()
print(f"Loaded {len(image_features)} image features.")

# --- Filter descriptions to only include images that have features ---
# This avoids errors if some images were skipped during extraction
all_image_ids = list(image_features.keys())
train_descriptions = {k: v for k, v in raw_descriptions.items() if k in all_image_ids}
print(f"Total training image-caption pairs: {sum(len(v) for v in train_descriptions.values())}")

print("\nReady to define and train the model.")


# --- 3. Define the Model Architecture ---
print("\n--- Defining Model Architecture ---")

# Call the function from model_architecture.py using the calculated sizes
# Feature dimension is dynamically extracted from the loaded data
feature_dim = image_features[all_image_ids[0]].shape[-1] 
model = create_attention_model(
    vocab_size=VOCAB_SIZE, 
    max_length=max_length, 
    feature_dim=feature_dim
)

print(model.summary())
print("Model definition complete.")

# -------------------------------------------------------------
# 4. FINAL STEP: Training the Model (This block runs immediately)
# -------------------------------------------------------------

# Determine the number of steps per epoch
# Steps_per_epoch = Total_Samples / Batch_Size
total_samples = sum(len(v) for v in train_descriptions.values())
steps = total_samples // BATCH_SIZE

# --- Callbacks for saving and stopping ---
# Saves the best model weights based on the lowest loss value
filepath = 'model-weights/attention-model-ep{epoch:03d}-loss{loss:.3f}.keras'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# Stops training if loss doesn't improve for 3 consecutive epochs
early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1, mode='min')

print("\n--- Starting Model Training ---")

# Create the data generator instance (it feeds data batches to the model)
generator = data_generator(train_descriptions, image_features, tokenizer, max_length, VOCAB_SIZE, BATCH_SIZE)

# Train the model
history = model.fit(
    generator,
    epochs=EPOCHS,
    steps_per_epoch=steps,
    verbose=1,
    callbacks=[checkpoint, early_stop]
)

print("\n--- Training Complete ---")
print("Best weights saved. Now ready for Inference (Caption Generation).")