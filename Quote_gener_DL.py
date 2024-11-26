import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
import os
import nltk
from collections import Counter
import json

# Ensure you have the NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
with open('quotes.json', 'r', encoding='utf-8') as file:
    quotes = json.load(file)

df = pd.DataFrame(quotes)

class DeepLearningQuoteGenerator:
    def __init__(self, df, seq_length=40, batch_size=128, epochs=1):
        self.df = df
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.chars = sorted(list(set(''.join(df['Quote']))))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.model = self.build_model()
        self.history = None

    def build_model(self):
        model = Sequential([
            LSTM(128, input_shape=(self.seq_length, self.vocab_size), return_sequences=True),
            Dropout(0.2),
            LSTM(128),
            Dropout(0.2),
            Dense(self.vocab_size // 2, activation='relu'),
            Dense(self.vocab_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def data_generator(self):
        text = ' '.join(self.df['Quote'])
        while True:
            for i in range(0, len(text) - self.seq_length, self.batch_size):
                X = np.zeros((self.batch_size, self.seq_length, self.vocab_size))
                y = np.zeros((self.batch_size, self.vocab_size))
                for j in range(self.batch_size):
                    if i + j + self.seq_length < len(text):
                        seq = text[i+j:i+j+self.seq_length]
                        next_char = text[i+j+self.seq_length]
                        for t, char in enumerate(seq):
                            X[j, t, self.char_to_idx[char]] = 1
                        y[j, self.char_to_idx[next_char]] = 1
                yield X, y

    def train(self):
        steps_per_epoch = len(' '.join(self.df['Quote'])) // self.batch_size
        self.history = self.model.fit(self.data_generator(), 
                                      steps_per_epoch=steps_per_epoch, 
                                      epochs=self.epochs,
                                      verbose=1)

    def generate_quote(self, seed_text, length=100):
        generated = seed_text
        for _ in range(length):
            x = np.zeros((1, self.seq_length, self.vocab_size))
            for t, char in enumerate(generated[-self.seq_length:]):
                if char in self.char_to_idx:
                    x[0, t, self.char_to_idx[char]] = 1
            
            preds = self.model.predict(x, verbose=0)[0]
            next_index = np.random.choice(len(preds), p=preds)
            next_char = self.idx_to_char[next_index]
            generated += next_char
            
            if next_char == '.':
                break
        
        return generated

    def plot_training_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'])
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, df):
        generator = cls(df)
        generator.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return generator

class QuoteGeneratorApp:
    def __init__(self, model_path, df):
        self.generator = DeepLearningQuoteGenerator.load_model(model_path, df)
        self.df = df

    def generate_multiple_quotes(self, count):
        for i in range(count):
            seed = random.choice(self.df['Quote'])[:40]
            new_quote = self.generator.generate_quote(seed)
            print(f"Quote {i+1}: {new_quote}")
            print()

    def interactive_quote_generation(self):
        while True:
            seed_text = input("Enter a seed text (or 'quit' to exit): ")
            if seed_text.lower() == 'quit':
                break
            generated_quote = self.generator.generate_quote(seed_text, length=100)
            print(f"Generated Quote: {generated_quote}")
            print()

    def analyze_quote(self, quote):
        words = nltk.word_tokenize(quote)
        pos_tags = nltk.pos_tag(words)
        return Counter(tag for word, tag in pos_tags)

    def generate_and_analyze_quote(self):
        seed = random.choice(self.df['Quote'])[:40]
        new_quote = self.generator.generate_quote(seed)
        analysis = self.analyze_quote(new_quote)
        print(f"Generated Quote: {new_quote}")
        print("Part of Speech Analysis:", analysis)

    def spawn_quotes_with_keywords(self, keywords, count=5):
        for i in range(count):
            seed = ' '.join(random.sample(keywords, k=min(3, len(keywords))))
            new_quote = self.generator.generate_quote(seed)
            print(f"Quote {i+1}: {new_quote}")
            print()

    def run(self):
        while True:
            print("\nQuote Generator App")
            print("1. Generate multiple quotes")
            print("2. Interactive quote generation")
            print("3. Generate and analyze a quote")
            print("4. Spawn quotes with keywords")
            print("5. Quit")
            
            choice = input("Enter your choice (1-5): ")
            
            if choice == '1':
                count = int(input("How many quotes do you want to generate? "))
                self.generate_multiple_quotes(count)
            elif choice == '2':
                self.interactive_quote_generation()
            elif choice == '3':
                self.generate_and_analyze_quote()
            elif choice == '4':
                keywords = input("Enter keywords separated by spaces: ").split()
                count = int(input("How many quotes do you want to generate? "))
                self.spawn_quotes_with_keywords(keywords, count)
            elif choice == '5':
                print("Thank you for using the Quote Generator App!")
                break
            else:
                print("Invalid choice. Please try again.")

# Usage
if __name__ == "__main__":
    # Assuming you have already trained and saved the model
    model_path = 'quote_generator_model.h5'
    
    # Load your DataFrame here
    # df = pd.read_csv('your_quotes_file.csv')
    
    app = QuoteGeneratorApp(model_path, df)
    app.run()
