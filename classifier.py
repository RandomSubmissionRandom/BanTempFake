from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import csv
import sys
from config import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
csv.field_size_limit(sys.maxsize)
def make_confusion_matrix(labels, preds, file_name, title='Confusion Matrix'):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_to_int.keys(), yticklabels=label_to_int.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

with open(dataset_file, encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f, quotechar='"')
    next(reader)
    rows = []
    for row in reader:
        if len(row) >= 2:
            text = row[0]
            label = row[1].strip()
            rows.append({'text': text, 'label': label})


# 70-30 Split
# MODEL 1: csebuetnlp/banglabert
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_1 = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
dataset = TextDataset(texts, labels, tokenizer_1)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_1)
val_dataset = TextDataset(X_val, y_val, tokenizer_1)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (csebuetnlp/banglabert, Epochs = 0, 70-30 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_1_30_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    batch_labels = batch['label'].to(device)
    outputs = model_1(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (csebuetnlp/banglabert, Epochs = 1, 70-30 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_1_30_split_epoch_1.png')

# MODEL 2
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_2 = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
dataset = TextDataset(texts, labels, tokenizer_2)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_2)
val_dataset = TextDataset(X_val, y_val, tokenizer_2)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_2 = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=2)
model_2.to(device)
optimizer = AdamW(model_2.parameters(), lr=2e-5)
model_2.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['label'].cpu().numpy()
        outputs = model_2(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (FacebookAI/xlm-roberta-base, Epochs = 0, 70-30 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_2_30_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_2 = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=2)
model_2.to(device)
optimizer = AdamW(model_2.parameters(), lr=2e-5)
model_2.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    batch_labels = batch['label'].to(device)
    outputs = model_2(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_2.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_2(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (FacebookAI/xlm-roberta-base, Epochs = 1, 70-30 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_2_30_split_epoch_1.png')

# 60-40 Split
# MODEL 1: csebuetnlp/banglabert
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_1 = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
dataset = TextDataset(texts, labels, tokenizer_1)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.4, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_1)
val_dataset = TextDataset(X_val, y_val, tokenizer_1)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (csebuetnlp/banglabert, Epochs = 0, 60-40 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_1_40_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model_1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (csebuetnlp/banglabert, Epochs = 1, 60-40 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_1_40_split_epoch_1.png')

# MODEL 2
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_2 = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
dataset = TextDataset(texts, labels, tokenizer_2)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.4, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_2)
val_dataset = TextDataset(X_val, y_val, tokenizer_2)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_2 = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=2)
model_2.to(device)
optimizer = AdamW(model_2.parameters(), lr=2e-5)
model_2.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_2(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (FacebookAI/xlm-roberta-base, Epochs = 0, 60-40 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_2_40_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_2 = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=2)
model_2.to(device)
optimizer = AdamW(model_2.parameters(), lr=2e-5)
model_2.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model_2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_2.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_2(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (FacebookAI/xlm-roberta-base, Epochs = 1, 60-40 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_2_40_split_epoch_1.png')

# 50-50 Split
# MODEL 1: csebuetnlp/banglabert
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_1 = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
dataset = TextDataset(texts, labels, tokenizer_1)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.5, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_1)
val_dataset = TextDataset(X_val, y_val, tokenizer_1)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (csebuetnlp/banglabert, Epochs = 0, 50-50 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_1_50_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model_1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (csebuetnlp/banglabert, Epochs = 1, 50-50 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_1_50_split_epoch_1.png')

# MODEL 2
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_2 = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
dataset = TextDataset(texts, labels, tokenizer_2)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.5, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_2)
val_dataset = TextDataset(X_val, y_val, tokenizer_2)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_2 = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=2)
model_2.to(device)
optimizer = AdamW(model_2.parameters(), lr=2e-5)
model_2.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_2(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (FacebookAI/xlm-roberta-base, Epochs = 0, 50-50 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_2_50_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_2 = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=2)
model_2.to(device)
optimizer = AdamW(model_2.parameters(), lr=2e-5)
model_2.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model_2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_2.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_2(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (FacebookAI/xlm-roberta-base, Epochs = 1, 50-50 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_2_50_split_epoch_1.png')

# MODEL 3: ai4bharat/indic-bert
# 70-30 Split
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_1 = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
dataset = TextDataset(texts, labels, tokenizer_1)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_1)
val_dataset = TextDataset(X_val, y_val, tokenizer_1)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (ai4bharat/indic-bert, Epochs = 0, 70-30 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_3_30_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model_1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (ai4bharat/indic-bert, Epochs = 1, 70-30 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_3_30_split_epoch_1.png')

# 60-40 Split
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_1 = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
dataset = TextDataset(texts, labels, tokenizer_1)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.4, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_1)
val_dataset = TextDataset(X_val, y_val, tokenizer_1)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (ai4bharat/indic-bert, Epochs = 0, 60-40 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_3_40_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model_1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (ai4bharat/indic-bert, Epochs = 1, 60-40 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_3_40_split_epoch_1.png')

# 50-50 Split
# Load the tokenizer and model
df = pd.DataFrame(rows)
texts = df['text'].tolist()
unique_labels = sorted(set(df['label'].tolist()))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
labels = [label_to_int[label] for label in df['label'].tolist()]
tokenizer_1 = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
dataset = TextDataset(texts, labels, tokenizer_1)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.5, random_state=42, stratify=labels)
train_dataset = TextDataset(X_train, y_train, tokenizer_1)
val_dataset = TextDataset(X_val, y_val, tokenizer_1)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (ai4bharat/indic-bert, Epochs = 0, 50-50 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_3_50_split_epoch_0.png')

# Epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2)
model_1.to(device)
optimizer = AdamW(model_1.parameters(), lr=2e-5)
model_1.train()
total_loss = 0
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model_1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
print(f"Training Loss: {total_loss/len(train_loader):.4f}")
model_1.eval()
all_preds = []
all_labels = []
val_losses = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        outputs = model_1(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.loss is not None:
            val_losses.append(outputs.loss.item())
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
val_loss = np.mean(val_losses)
print(f"Validation Loss: {val_loss:.4f}")
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
print("Test Results (ai4bharat/indic-bert, Epochs = 1, 50-50 split):")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(classification_report(all_labels, all_preds, target_names=list(label_to_int.keys()), output_dict=True))
make_confusion_matrix(all_labels, all_preds, 'confusion_matrix_model_3_50_split_epoch_1.png')
