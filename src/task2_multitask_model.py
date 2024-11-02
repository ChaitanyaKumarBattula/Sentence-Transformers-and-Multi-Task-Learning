import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Load pretrained model and tokenizer
# Model: sentence-transformers/all-MiniLM-L6-v2, chosen for efficiency in embedding
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

# Define Multi-Task Model with Separate Heads for Each Task
class MultiTaskTransformer(nn.Module):
    def __init__(self, encoder, num_classes_task_a=3, num_classes_task_b=3):
        super(MultiTaskTransformer, self).__init__()
        self.encoder = encoder

         # Task A: Sentence Classification
        self.classification_head = nn.Linear(384, num_classes_task_a)  
        
        # Task B: Sentiment Analysis
        self.sentiment_head = nn.Linear(384, num_classes_task_b)       

    def forward(self, input_ids, attention_mask):
        # Freeze the encoder during forward pass for efficiency
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling for fixed-length embedding
        pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)

        # Task-specific outputs for classification and sentiment analysis
        classification_logits = self.classification_head(pooled_output)
        sentiment_logits = self.sentiment_head(pooled_output)

        return classification_logits, sentiment_logits

# Test code to initialize and test the model
if __name__ == "__main__":
    # Initialize the multi-task model with the encoder and task-specific heads
    multi_task_model = MultiTaskTransformer(encoder)
    sample_sentence = [
    "I absolutely love the taste of this pizza!",
    "This dress is stunning and fits perfectly.",
    "The service was terrible, I am very disappointed."]

    # sample_sentence = ["I am excited to learn about transformers."]
    inputs = tokenizer(sample_sentence, return_tensors="pt", truncation=True, padding=True)
    classification_output, sentiment_output = multi_task_model(inputs['input_ids'], inputs['attention_mask'])
    print("Task A (Sentence Classification) Output:", classification_output)
    print("Task B (Sentiment Analysis) Output:", sentiment_output)

