import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from task3_train import initialize_model, get_loss_fn, train_model

# Initialize model and tokenizer
model, tokenizer = initialize_model()
loss_fn = get_loss_fn()

# Define layer-wise learning rates
learning_rates = {
    'encoder.embeddings': 1e-6,
    'encoder.encoder.layer': 1e-5,
    'classification_head': 1e-4,
    'sentiment_head': 1e-4,
}

# Custom optimizer with layer-wise learning rates
param_groups = []
for name, param in model.named_parameters():
    if 'encoder.embeddings' in name:
        param_groups.append({'params': param, 'lr': learning_rates['encoder.embeddings']})
    elif 'encoder.encoder.layer' in name:
        param_groups.append({'params': param, 'lr': learning_rates['encoder.encoder.layer']})
    elif 'classification_head' in name:
        param_groups.append({'params': param, 'lr': learning_rates['classification_head']})
    elif 'sentiment_head' in name:
        param_groups.append({'params': param, 'lr': learning_rates['sentiment_head']})

optimizer = optim.Adam(param_groups)

# Function to create a DataLoader with dummy data
def get_data_loader(tokenizer):
    # Dummy data (replace with actual dataset as needed)
    sentences = ["I am happy", "This is a great day", "I am sad"]
    labels_task_a = torch.tensor([0, 1, 2])  # Example labels for Task A
    labels_task_b = torch.tensor([1, 0, 2])  # Example labels for Task B

    # Tokenize sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Create DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels_task_a, labels_task_b)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    return data_loader

# Get data loader with dummy data
data_loader = get_data_loader(tokenizer)

# Train with layer-wise learning rate
train_model(model, optimizer, loss_fn, data_loader, epochs=3)
