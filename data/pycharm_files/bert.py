import pandas as pd
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, pipeline
from sklearn.model_selection import train_test_split
import evaluate
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("C:/Users/Ravi Kiran/Downloads/dataset/case_study_data.csv")  # Update with the actual path to your CSV

# ✅ Reduce dataset size for faster training (Optional, can increase later)
df = df.sample(5000, random_state=42)

# Encode labels
label_encoder = LabelEncoder()
df['product_group_numeric'] = label_encoder.fit_transform(df['product_group'])


# Split dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(df['text'], df['product_group_numeric'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 50% of temp set

# Convert to DataFrames
train_df = pd.DataFrame({'text': X_train, 'label': y_train}).reset_index(drop=True)
val_df = pd.DataFrame({'text': X_val, 'label': y_val}).reset_index(drop=True)
test_df = pd.DataFrame({'text': X_test, 'label': y_test}).reset_index(drop=True)

# Convert labels to integers
train_df['label'] = train_df['label'].astype(int)
val_df['label'] = val_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

# Convert to Hugging Face Dataset format
train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)
test_data = Dataset.from_pandas(test_df)

# Store in DatasetDict
dataset = DatasetDict({'train': train_data, 'validation': val_data, 'test': test_data})

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenization function
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized_inputs["label"] = examples["label"]  # Ensure labels are included
    return tokenized_inputs

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Convert labels to torch tensors
tokenized_datasets = tokenized_datasets.map(lambda x: {"label": torch.tensor(x["label"]).long()}, batched=True)

# Load pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(df['product_group_numeric'].unique()))

# Define evaluation metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Define training arguments
training_args_cls = TrainingArguments(
    output_dir="./roberta_classification",
    evaluation_strategy="epoch",  # ✅ Ensures evaluation happens after each epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer_cls = Trainer(
    model=model,
    args=training_args_cls,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # ✅ Fix: Include validation dataset
    compute_metrics=compute_metrics,
)

# Train the model
trainer_cls.train()

# Save the trained model and tokenizer
model.save_pretrained("./roberta_classification")
tokenizer.save_pretrained("./roberta_classification")

# Load the trained model for inference
classifier_model = RobertaForSequenceClassification.from_pretrained("./roberta_classification")
classifier_tokenizer = RobertaTokenizer.from_pretrained("./roberta_classification")

# Create classifier pipeline
classifier = pipeline("text-classification", model=classifier_model, tokenizer=classifier_tokenizer)

# Test predictions
for text in X_test[:5]:
    print(f"Text: {text}\nPrediction: {classifier(text)}\n")

print(y_test[:5])  # ✅ Fix: Ensure it's outside the loop