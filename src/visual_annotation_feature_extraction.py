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
        indices = torch.arange(batch_size, device=last_hidden_states.device)
        return last_hidden_states[indices, sequence_lengths]


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
        embeddings = last_token_pool(
            outputs.last_hidden_state,
            batch_dict['attention_mask']
        )
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def load_annotation_files(data_dir):
    """Load all annotation text files from directory"""
    data_path = Path(data_dir)
    annotation_files = list(data_path.glob("*.txt"))

    print(f"Found {len(annotation_files)} annotation files")

    annotations = []
    file_names = []

    for file_path in annotation_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                annotation = f.read().strip()
                if annotation:
                    annotations.append(annotation)
                    file_names.append(file_path.name)
                    print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return annotations, file_names


def save_annotation_features(
    embeddings, file_names, output_dir='output/features/'
):
    """Save extracted annotation features to JSONL file"""

    os.makedirs(output_dir, exist_ok=True)

    jsonl_file = os.path.join(output_dir, 'visual_annotation_features.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for file_name, embedding in zip(file_names, embeddings):
            record = {
                'file_name': file_name,
                'embedding': embedding.flatten().tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {len(file_names)} annotation embeddings to {jsonl_file}")

    np_file = os.path.join(output_dir, 'visual_annotation_features.npy')
    np.save(np_file, embeddings)
    print(
        f"Saved annotation embeddings array: {embeddings.shape} to {np_file}"
    )

    metadata = {
        'file_names': file_names,
        'embedding_shape': embeddings.shape,
        'embedding_dim': embeddings.shape[1],
        'num_files': len(file_names)
    }

    metadata_file = os.path.join(output_dir, 'visual_annotation_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Annotation features saved to {output_dir}")


def main():
    """Main function to extract embeddings from image annotations"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    annotations_dir = project_root / "data" / "images" / "annotations"
    model_path = project_root / "models" / "Qwen3-Embedding-0.6B"

    # Load Qwen3 embedding model
    model, tokenizer = load_qwen_embedding_model(model_path)
    model = model.to(device)

    # Load annotation files
    print("\nLoading annotation files...")
    annotations, file_names = load_annotation_files(annotations_dir)

    if not annotations:
        print(f"No annotation files found in {annotations_dir}")
        return

    print(f"\nExtracting embeddings from {len(annotations)} annotations...")

    # Extract embeddings
    embeddings = []
    for annotation in tqdm(annotations, desc="Extracting embeddings"):
        embedding = extract_embeddings(annotation, model, tokenizer, device)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    print("\nEmbedding shapes:")
    print(f"Number of annotations: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Save features
    print("\nSaving annotation features...")
    output_dir = project_root / "output" / "features"
    save_annotation_features(embeddings, file_names, output_dir)

    print("\nVisual annotation feature extraction complete!")


if __name__ == "__main__":
    main()
