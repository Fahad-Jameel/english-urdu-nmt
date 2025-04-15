
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load tokenizers
with open('models/eng_tokenizer.pickle', 'rb') as handle:
    eng_tokenizer = pickle.load(handle)
    
with open('models/urdu_tokenizer.pickle', 'rb') as handle:
    urdu_tokenizer = pickle.load(handle)

# Load models
encoder_model = load_model('models/encoder_model.h5')
decoder_model = load_model('models/decoder_model.h5')

# Maximum sequence lengths (from training)
max_eng_length = 50
max_urdu_length = 50

def preprocess_english(text):
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Add start and end tokens
    return '<start> ' + text + ' <end>'

def translate_sentence(input_sentence):
    # Preprocess input sentence
    input_sentence = preprocess_english(input_sentence)
    
    # Convert to sequence
    input_seq = eng_tokenizer.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_eng_length, padding='post')
    
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Generate empty target sequence with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = urdu_tokenizer.word_index['<start>']
    
    # Output sequence
    decoded_sentence = ''
    
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_word = ''
        for word, index in urdu_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_urdu_length:
            stop_condition = True
        elif sampled_word != '<start>':  # Don't add start token to output
            decoded_sentence += ' ' + sampled_word
        
        # Update the target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence.strip()

def interactive_translation():
    print("\n=== English to Urdu Translation Demo ===")
    print("Type 'quit' to exit the demo")
    
    while True:
        user_input = input("\nEnter English text to translate: ")
        if user_input.lower() == 'quit':
            break
            
        translation = translate_sentence(user_input)
        print(f"Urdu translation: {translation}")

# Run the interactive translation demo
if __name__ == "__main__":
    interactive_translation()
