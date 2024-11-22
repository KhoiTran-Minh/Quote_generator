import tkinter as tk
from tkinter import ttk
import random
import pandas as pd
import json
import pandas as pd

# Read the JSON file
with open('quotes.json', 'r', encoding='utf-8') as file:
    quotes = json.load(file)

df = pd.DataFrame(quotes) #Convert the JSON data to a dataFrame

import random
import pandas as pd

class DetailedQuoteGenerator:
    def __init__(self, df):
        self.df = df
        self.authors = df['Author'].unique()
        self.tags = [tag for tags in df['Tags'] for tag in tags]
        self.categories = df['Category'].unique()

    def generate_quote(self):

        # Randomly select a row from the DataFrame
        quote_row = self.df.sample(n=1).iloc[0]

        # Generate a new quote using Markov Chain-like approach
        words = quote_row['Quote'].split()
        new_quote = self.generate_markov_quote(words)

        # Use the selected row's author and category
        author = quote_row['Author']
        category = quote_row['Category']

        # Randomly select one tag from the quote's tags
        tag = random.choice(quote_row['Tags'])

        # Generate a random popularity score
        popularity = random.randint(1, 100)

        return {
            'Quote': new_quote,
            'Author': author,
            'Tags': tag,
            'Popularity': popularity,
            'Category': category
        }

    def generate_markov_quote(self, words, max_length=30):
        markov_dict = {}
        for i in range(len(words) - 1):
            if words[i] not in markov_dict:
                markov_dict[words[i]] = []
            markov_dict[words[i]].append(words[i + 1])

        current_word = random.choice(words)
        result = [current_word]

        while len(result) < max_length:
            if current_word not in markov_dict:
                break
            next_word = random.choice(markov_dict[current_word])
            result.append(next_word)
            current_word = next_word

        return ' '.join(result)

class QuoteGeneratorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Quote Generator")
        self.master.geometry("600x400")

        self.generator = DetailedQuoteGenerator(df)

        self.create_widgets()

    def create_widgets(self):
        # Generate button
        self.generate_button = ttk.Button(self.master, text="Generate Quote", command=self.generate_and_display_quote)
        self.generate_button.pack(pady=20)

        # Frame for quote details
        self.quote_frame = ttk.Frame(self.master, padding="10")
        self.quote_frame.pack(fill=tk.BOTH, expand=True)

        # Labels for quote details
        self.quote_label = ttk.Label(self.quote_frame, text="", wraplength=550, justify="center")
        self.quote_label.pack(pady=10)

        self.author_label = ttk.Label(self.quote_frame, text="")
        self.author_label.pack()

        self.tag_label = ttk.Label(self.quote_frame, text="")
        self.tag_label.pack()

        self.popularity_label = ttk.Label(self.quote_frame, text="")
        self.popularity_label.pack()

        self.category_label = ttk.Label(self.quote_frame, text="")
        self.category_label.pack()

    def generate_and_display_quote(self):
        new_quote = self.generator.generate_quote()
        
        self.quote_label.config(text=f'"{new_quote["Quote"]}"')
        self.author_label.config(text=f"Author: {new_quote['Author']}")
        self.tag_label.config(text=f"Tag: {new_quote['Tags']}")
        self.popularity_label.config(text=f"Popularity: {new_quote['Popularity']}")
        self.category_label.config(text=f"Category: {new_quote['Category']}")

# Create the main window and run the app
root = tk.Tk()
app = QuoteGeneratorApp(root)
root.mainloop()
