<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <h1>Sentiment Analysis with Multilingual Support</h1>

  <p>This project aims to perform sentiment analysis on text data with multilingual support using deep learning techniques. The model is trained to classify text into positive or negative sentiments and can handle input text in various languages.</p>

  <h2>Dataset</h2>

  <p>The sentiment analysis model is trained on the <a href="https://www.kaggle.com/kazanova/sentiment140">Sentiment Analysis Dataset</a>, which contains 1.6 million tweets labeled with positive or negative sentiments. The dataset is preprocessed to remove noise, such as special characters, URLs, emails, and numbers. Texts are tokenized, lemmatized, and cleaned from stopwords before feeding into the model.</p>

  <h2>Model Architecture</h2>

  <p>The sentiment analysis model is built using TensorFlow and Keras. It utilizes a transformer-based architecture, specifically a Transformer Encoder, to capture the contextual information of the input text. The model consists of an embedding layer, followed by a Transformer Encoder layer, a global max-pooling layer, dropout layers for regularization, and a dense layer with softmax activation for classification.</p>

  <h2>Training and Evaluation</h2>

  <p>The model is trained using a subset of the dataset split into training and validation sets. Training is conducted for a specified number of epochs, and model performance is evaluated using accuracy as the metric. The trained model achieves high accuracy on the validation set, indicating its effectiveness in sentiment classification.</p>

  <h2>Multilingual Support</h2>

  <p>The model supports text input in multiple languages through translation to English using the Helsinki-NLP Opus MT model. Texts are translated to English before sentiment analysis to ensure consistent processing. Sample texts in French, Spanish, German, and Italian are provided to demonstrate the model's multilingual capability.</p>

  <h2>Usage</h2>

  <ol>
    <li>Install the required dependencies listed in <code>requirements.txt</code>.</li>
    <li>Load the trained model and tokenizer configuration.</li>
    <li>Preprocess input text by translating it to English and padding sequences.</li>
    <li>Make predictions on the preprocessed text to obtain sentiment labels and probabilities.</li>
  </ol>

  <h2>Example</h2>

  <pre><code>
tokenizer_config = tokenizer.to_json()
with open("tokenizer_config.json", "w") as tokenizer_file:
    tokenizer_file.write(tokenizer_config)

# Define a function to predict sentiment for text in any language
def predict_sentiment(text, src_lang):
    # Translate text to English
    english_text = translate_to_english(text, src_lang)
    
    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([english_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=sequence_length, padding='post')
    
    # Make prediction
    predicted_probabilities = model.predict(padded_sequence)
    predicted_class = np.argmax(predicted_probabilities)
    
    # Map predicted class to sentiment label
    sentiment_label = "Positive" if predicted_class == 1 else "Negative"
    
    return sentiment_label, predicted_probabilities

# Define a sample text in different languages
sample_texts = {
    'fr': 'Je suis heureux.',
    'es': 'Estoy feliz.',
    'de': 'Ich bin glücklich.',
    'it': 'Sono felice.'
}

# Predict sentiment for each sample text
for lang, text in sample_texts.items():
    sentiment, probabilities = predict_sentiment(text, lang)
    print(f"Input ({lang}): {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Predicted Probabilities: {probabilities}")
    print()
  </code></pre>

  <h2>Acknowledgments</h2>

  <ul>
    <li><a href="https://www.kaggle.com/">Kaggle</a> for providing the Sentiment Analysis Dataset.</li>
    <li><a href="https://huggingface.co/Helsinki-NLP">Helsinki-NLP</a> for the Opus MT model.</li>
  </ul>

  <h2>License</h2>

  <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>
</html>
