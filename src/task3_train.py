import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from task2_multitask_model import MultiTaskTransformer

# Model and tokenizer initialization
def initialize_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    model = MultiTaskTransformer(encoder, num_classes_task_a=3, num_classes_task_b=3)
    return model, tokenizer

# Loss function
def get_loss_fn():
    return nn.CrossEntropyLoss()

# Optimizer for Task 3 (can be replaced for Task 4)
def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop function
def train_model(model, optimizer, loss_fn, data_loader, epochs=3):
    print("Starting training...")  # Debugging check
    for epoch in range(epochs):
        model.train()
        total_loss_task_a, total_loss_task_b = 0, 0
        for batch in data_loader:
            input_ids, attention_mask, labels_a, labels_b = batch

            # Forward pass
            classification_logits, sentiment_logits = model(input_ids, attention_mask)

            # Calculate losses for each task
            loss_task_a = loss_fn(classification_logits, labels_a)
            loss_task_b = loss_fn(sentiment_logits, labels_b)
            total_loss = loss_task_a + loss_task_b

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss_task_a += loss_task_a.item()
            total_loss_task_b += loss_task_b.item()

        # Print losses for the epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss Task A: {total_loss_task_a:.4f}, Loss Task B: {total_loss_task_b:.4f}")

    print("Training completed.")

# Set up data loader with dummy data (replace with real dataset as needed)
def get_data_loader(tokenizer):
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

# Main function to run the training
if __name__ == "__main__":
    model, tokenizer = initialize_model()
    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()
    data_loader = get_data_loader(tokenizer)

    # Run the training loop
    train_model(model, optimizer, loss_fn, data_loader, epochs=3)
