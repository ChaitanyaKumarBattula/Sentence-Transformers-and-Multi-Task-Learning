from transformers import AutoTokenizer, AutoModel
import torch

# Load pretrained model and tokenizer
# Model: sentence-transformers/all-MiniLM-L6-v2 - chosen for efficiency and general-purpose sentence embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Encode sentences into embeddings
def encode_sentence(sentence):
    """
    Encodes a sentence into a fixed-length embedding using mean pooling.
    
    Parameters:
        sentence (str): Input sentence to encode.
    
    Returns:
        torch.Tensor: Fixed-length embedding for the input sentence.
    """

    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

     # Generate hidden states from the model without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling to get fixed-length embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Test with a few sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Deep learning models are powerful.",
    "Machine learning is transforming industries."
]

# Generate and display embeddings for each sentence
for sentence in sentences:
    embedding = encode_sentence(sentence)
    print(f"Embedding for '{sentence}':")
    print(f"Shape: {embedding.shape}")
    print(f"Sample values: {embedding[0, :5]}\n")
    