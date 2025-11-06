import torch
import torch.nn.functional as F
import json
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states, attention_mask):
    """Pool the last token from the hidden states"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def load_qwen_embedding_model(model_path):
    """Load Qwen3 embedding model"""
    print(f"Loading Qwen3 embedding model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    return model, tokenizer


def extract_embeddings(text, model, tokenizer, device, max_length=8192):
    """Extract embeddings from text using Qwen3 model"""

    # Tokenize input text
    batch_dict = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**batch_dict)
        # Use last token pooling as per official example
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def load_text_files(data_dir):
    """Load all text files from directory"""
    data_path = Path(data_dir)
    text_files = list(data_path.glob("*.txt"))

    print(f"Found {len(text_files)} text files")

    texts = []
    file_names = []

    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    texts.append(text)
                    file_names.append(file_path.name)
                    print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return texts, file_names


def save_text_features(embeddings, file_names, output_dir='output/features/'):
    """Save extracted text features to JSONL file"""

    os.makedirs(output_dir, exist_ok=True)

    jsonl_file = os.path.join(output_dir, 'textual_features.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i, (file_name, embedding) in enumerate(zip(file_names, embeddings)):
            record = {
                'file_name': file_name,
                'embedding': embedding.flatten().tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {len(file_names)} text embeddings to {jsonl_file}")

    np_file = os.path.join(output_dir, 'textual_features.npy')
    np.save(np_file, embeddings)
    print(f"Saved text embeddings array: {embeddings.shape} to {np_file}")

    metadata = {
        'file_names': file_names,
        'embedding_shape': embeddings.shape,
        'embedding_dim': embeddings.shape[1],
        'num_files': len(file_names)
    }

    with open(os.path.join(output_dir, 'textual_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Text features saved to {output_dir}")


def main():
    """Main function to extract text embeddings"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Qwen3 embedding model
    model_path = "models/Qwen3-Embedding-0.6B"
    model, tokenizer = load_qwen_embedding_model(model_path)
    model = model.to(device)

    # Load text files
    print("Loading text files...")
    texts, file_names = load_text_files('data/text/')

    if not texts:
        print("No text files found in data/text/ directory")
        return

    print(f"Extracting embeddings from {len(texts)} text files...")

    # Extract embeddings
    embeddings = []
    for i, text in enumerate(tqdm(texts, desc="Extracting embeddings")):
        embedding = extract_embeddings(text, model, tokenizer, device)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    print(f"\nEmbedding shapes:")
    print(f"Number of texts: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Save features
    print("\nSaving text features...")
    save_text_features(embeddings, file_names)

    print("Text feature extraction complete!")


if __name__ == "__main__":
    main()
