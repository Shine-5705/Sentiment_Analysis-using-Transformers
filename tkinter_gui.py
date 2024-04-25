import tkinter as tk
from transformers import pipeline, AutoTokenizer
import pandas as pd

translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")
sentiment_analysis_pipeline = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")

def translate_and_analyze():
    input_statements = entry_statements.get("1.0", "end-1c").splitlines()
    input_languages = entry_languages.get("1.0", "end-1c").splitlines()
    
    data = {
        "tweet": input_statements,
        "language": input_languages,
    }
    
    df = pd.DataFrame(data)
    df["sentiment_label"], df["sentiment_score"] = zip(*df["tweet"].apply(preprocess_data))
    
    result_text.delete("1.0", "end")
    result_text.insert("end", df.to_string(index=False))

def preprocess_data(text):
    translated_text = translation_pipeline(text, src_lang="auto", tgt_lang="en")[0]['translation_text']
    encoded_input = tokenizer(translated_text, return_tensors="tf", padding=True, truncation=True)
    input_text = tokenizer.decode(encoded_input['input_ids'][0])
    sentiment_result = sentiment_analysis_pipeline(input_text)[0]
    sentiment_label = sentiment_result["label"]
    sentiment_score = sentiment_result["score"]
    return sentiment_label, sentiment_score

# GUI Setup
window = tk.Tk()
window.title("Sentiment Analysis")

label_statements = tk.Label(window, text="Input Statements:")
label_statements.grid(row=0, column=0, padx=10, pady=5, sticky="w")

entry_statements = tk.Text(window, height=5, width=50)
entry_statements.grid(row=0, column=1, padx=10, pady=5)

label_languages = tk.Label(window, text="Languages:")
label_languages.grid(row=1, column=0, padx=10, pady=5, sticky="w")

entry_languages = tk.Text(window, height=5, width=50)
entry_languages.grid(row=1, column=1, padx=10, pady=5)

button_analyze = tk.Button(window, text="Analyze", command=translate_and_analyze)
button_analyze.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

label_result = tk.Label(window, text="Sentiment Analysis Results:")
label_result.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")

result_text = tk.Text(window, height=10, width=70)
result_text.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

window.mainloop()
