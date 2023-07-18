import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random

random.seed(1)
# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Define the labels for the topics
topic_labels = ['Sports', 'Politics', 'Science']

# Input text to classify
input_text = "Physics is very interesting."

# Tokenize input text
input_tokens = tokenizer.tokenize(input_text)

# Add special tokens and convert to input IDs
input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + input_tokens + ['[SEP]'])

# Create input tensor
input_tensor = torch.tensor([input_ids])

# Perform classification
model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
    logits = outputs.logits

# Get predicted label
predicted_label_idx = logits.argmax().item()
predicted_label = topic_labels[predicted_label_idx]

# Print the predicted label
print(input_text)
print(topic_labels)
print("Predicted Topic Label:", predicted_label)
