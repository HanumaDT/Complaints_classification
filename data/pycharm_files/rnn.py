import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load dataset
data_path = "C:/Users/Ravi Kiran/Downloads/dataset/case_study_data.csv"
df = pd.read_csv(data_path)

# Encode labels
label_encoder = LabelEncoder()
df['product_group_numeric'] = label_encoder.fit_transform(df['product_group'])

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['clean_text'] = df['text'].apply(clean_text)

# 🟢 Reduce max_features from 5000 → 2000 to save memory
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_sparse = vectorizer.fit_transform(df['clean_text'])  # Keep as sparse matrix

# Convert to float32 to reduce memory
X = X_sparse.astype(np.float32).toarray()  # Convert sparse to dense with float32
y = df['product_group_numeric']  # Use numeric labels

# 🟢 Reduce dataset size (optional, use only 50% of the data)
X, _, y, _ = train_test_split(X, y, test_size=0.5, random_state=42)

# Split dataset into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for RNN (samples, timesteps=1, features)
X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_rnn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define RNN model
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(1, X_train.shape[1])),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile model
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
rnn_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test))

# Evaluate model
rnn_loss, rnn_acc = rnn_model.evaluate(X_test_rnn, y_test)
print(f"RNN Model Accuracy: {rnn_acc}")