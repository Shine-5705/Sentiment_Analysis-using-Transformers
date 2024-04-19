import tkinter as tk
from transformers import pipeline
import tensorflow as tf
import json
import numpy as np

translation_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Load the saved model and tokenizer
model = tf.saved_model.load("sentiment_analysis_model")
with open("tokenizer_config.json", "r") as tokenizer_file:
    tokenizer_config = tokenizer_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

def translate_to_english(text, src_lang):
    translated_text = translation_pipe(text, src_lang=src_lang, tgt_lang='en')
    return translated_text[0]['translation_text']

def preprocess_text(text):
    translated_text = translate_to_english(text, src_lang="auto")
    sequence = tokenizer.texts_to_sequences([translated_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=500, padding='post')
    return padded_sequence

def predict_sentiment(input_text):
    preprocessed_text = preprocess_text(input_text)
    predicted_probabilities = model(preprocessed_text)
    predicted_class = np.argmax(predicted_probabilities)
    sentiment_label = "Positive" if predicted_class == 1 else "Negative"
    return sentiment_label, predicted_probabilities

def analyze_sentiment():
    input_text = input_text_entry.get()
    sentiment, probabilities = predict_sentiment(input_text)
    sentiment_label.config(text=f"Predicted Sentiment: {sentiment}")
    probability_label.config(text=f"Probability (Positive): {probabilities[0][1]}")

# GUI
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("400x200")

input_text_label = tk.Label(root, text="Enter text:")
input_text_label.pack()

input_text_entry = tk.Entry(root, width=50)
input_text_entry.pack()

analyze_button = tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack()

sentiment_label = tk.Label(root, text="")
sentiment_label.pack()

probability_label = tk.Label(root, text="")
probability_label.pack()

root.mainloop()
