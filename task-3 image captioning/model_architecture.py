## model_architecture.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Attention
from tensorflow.keras.layers import Dropout, Concatenate, Layer, Activation, TimeDistributed
from tensorflow.keras import backend as K

# --- 1. Custom Bahdanau Attention Layer ---
# This is a custom layer implementation for Bahdanau Attention 
# (often integrated into newer Keras/TensorFlow versions, but good practice to define)

class BahdanauAttention(Layer):
    """
    Custom Bahdanau Attention Layer (Additive Attention)
    """
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        # input_shape[0] is the encoder output (image features)
        # input_shape[1] is the decoder hidden state (LSTM output)
        super(BahdanauAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs[0] = encoder_output (E) [Batch_Size, Max_Features, Feature_Dim] e.g., (64, 64, 2048)
        # inputs[1] = decoder_state (H) [Batch_Size, LSTM_Units] e.g., (64, 512)
        encoder_output, decoder_state = inputs
        
        # Expand decoder state (H) to match E's time dimension: [B, 1, LSTM_Units]
        decoder_state_expanded = K.expand_dims(decoder_state, 1)

        # Score calculation: V^T * tanh(W1*E + W2*H)
        score = self.V(K.tanh(self.W1(encoder_output) + self.W2(decoder_state_expanded)))
        
        # Attention weights (alpha): softmax(score) -> [B, Max_Features, 1]
        attention_weights = K.softmax(score, axis=1)

        # Context vector (c): sum(alpha * E) -> [B, Feature_Dim]
        context_vector = K.sum(attention_weights * encoder_output, axis=1)

        return context_vector, attention_weights

# --- 2. Complete Attention Model Architecture ---

def create_attention_model(vocab_size, max_length, feature_dim=2048, units=512):
    """
    Defines the complete Attention-based Image Captioning Model.
    """
    # ----------------------------------------------------
    # PART 1: ENCODER INPUT (Image Features)
    # ----------------------------------------------------
    
    # Input 1: Image Features (from ResNet output)
    # Shape: (Max_Features, Feature_Dim) e.g., (49 or 64, 2048)
    encoder_input = Input(shape=(None, feature_dim), name='encoder_input')
    
    # Simple Dense layer to compress/map features to a manageable size (units)
    encoder_output = TimeDistributed(Dense(units, activation='relu'))(encoder_input)

    # ----------------------------------------------------
    # PART 2: DECODER INPUT (Text Sequences)
    # ----------------------------------------------------
    
    # Input 2: Text Sequences (Padded Captions)
    # Shape: (Max_Length) e.g., (34)
    decoder_input = Input(shape=(max_length,), name='decoder_input')
    
    # Embedding Layer: converts integer IDs to dense vectors
    decoder_embedding = Embedding(vocab_size, units, mask_zero=True)(decoder_input)
    
    # LSTM Decoder: returns sequences (for attention) and state (for attention)
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name='lstm_decoder')
    lstm_output, state_h, state_c = decoder_lstm(decoder_embedding)
    
    # ----------------------------------------------------
    # PART 3: ATTENTION MECHANISM
    # ----------------------------------------------------
    
    # The Bahdanau Attention layer takes the Encoder Output (features) and the Decoder State (LSTM hidden state)
    attention_layer = BahdanauAttention(units=units, name='attention_mechanism')
    
    # We use the final state_h of the LSTM to calculate attention for the next time step
    context_vector, attention_weights = attention_layer([encoder_output, state_h])

    # ----------------------------------------------------
    # PART 4: CONCATENATION AND PREDICTION
    # ----------------------------------------------------
    
    # Combine the Context Vector and the LSTM output
    decoder_combined_input = Concatenate(axis=-1)([context_vector, state_h])

    # Final Dense layers for prediction
    output_dense1 = Dense(units, activation='relu')(decoder_combined_input)
    # Final layer: Softmax activation over the entire vocabulary size
    output_dense2 = Dense(vocab_size, activation='softmax')(output_dense1)

    # Define the final model
    model = Model(inputs=[encoder_input, decoder_input], outputs=output_dense2)
    
    # Compile the model
    # We use Categorical Crossentropy because the target (y) is one-hot encoded
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model