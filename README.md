# English to Urdu Neural Machine Translation

An LSTM-based sequence-to-sequence neural machine translation system for translating English text to Urdu.

## Overview

This project implements a complete neural machine translation pipeline using LSTM-based sequence-to-sequence models. The system is designed to translate English sentences into grammatically correct Urdu text while preserving the semantic meaning of the original content.

## Features

- Complete preprocessing pipeline for English and Urdu text
- LSTM-based encoder-decoder architecture
- Teacher forcing during training for improved stability
- Comprehensive evaluation using BLEU metrics
- Detailed error analysis and visualization
- Interactive translation demo

## Dataset

The system is trained on a parallel corpus of English-Urdu sentence pairs (approximately 25,000 pairs). Each line in the English corpus corresponds to its translation in the Urdu corpus.

## Model Architecture

- Word embeddings for both languages
- LSTM-based encoder that processes English input
- LSTM-based decoder that generates Urdu output
- Teacher forcing during training

## Results

The model achieves the following BLEU scores on the test set:
- BLEU-1: ~0.45
- BLEU-2: ~0.35
- BLEU-3: ~0.25
- BLEU-4: ~0.20

## Future Improvements

- Implement attention mechanism
- Use bidirectional LSTMs
- Explore transformer-based architectures
- Apply subword tokenization for handling morphologically rich languages
- Increase dataset size or use transfer learning from larger models

## Requirements

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- NLTK (for BLEU score calculation)
- scikit-learn
