import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data_path = "C:/Users/Ravi Kiran/Downloads/dataset/case_study_data.csv"
df = pd.read_csv(data_path)

# Encode labels
label_encoder = LabelEncoder()
df['product_group_numeric'] = label_encoder.fit_transform(df['product_group'])

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF vectorization (Optimized)
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')  # Reduce feature size
X = vectorizer.fit_transform(df['clean_text']).astype(np.float32)  # Keep sparse format & use float32
y = df['product_group_numeric']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert sparse matrix to dense AFTER splitting (Reduces memory pressure)
X_train = X_train.toarray()
X_test = X_test.toarray()

# Reshape for LSTM (samples, timesteps=1, features)
X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_rnn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1])),  # Fix: Use shape[1] instead of shape[2]
    Dense(len(np.unique(y)), activation='softmax')  # Number of classes
])

# Compile model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
lstm_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test))

# Evaluate model
lstm_loss, lstm_acc = lstm_model.evaluate(X_test_rnn, y_test)
print(f"LSTM Model Accuracy: {lstm_acc}")