from transformers import Dinov2Model, AutoImageProcessor
import torch
import numpy as np
from PIL import Image
import os
import glob
import json


def extract_features(
    images, model, processor,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Extract different types of features from DINOv2"""

    # Move model to device
    model = model.to(device)

    # Process images
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # CLS token for clustering
    cls_token = outputs.pooler_output.cpu().numpy()

    # L2 normalize embeddings
    norms = np.linalg.norm(cls_token, axis=1, keepdims=True)
    cls_token_normalized = cls_token / (norms + 1e-8)

    features = {
        'cls_token': cls_token_normalized
    }

    return features


def load_and_process_images(data_dir='data/', max_images=None):
    """Load all PNG images from data directory"""

    # Find all PNG files
    image_paths = glob.glob(os.path.join(data_dir, '*.png'))

    if max_images:
        image_paths = image_paths[:max_images]

    print(f"Found {len(image_paths)} PNG images")

    images = []
    image_names = []

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            images.append(image)
            image_names.append(os.path.basename(img_path))
            print(f"Loaded: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return images, image_names


def save_features(features, image_names, output_dir='output/features/'):
    """Save extracted features to files"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSONL (one embedding per line)
    cls_token_data = features['cls_token']

    jsonl_file = os.path.join(output_dir, 'visual_features.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for image_name, embedding in zip(image_names, cls_token_data):
            record = {
                'image_name': image_name,
                'embedding': embedding.tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {len(image_names)} embeddings to {jsonl_file}")

    np_file = os.path.join(output_dir, 'visual_features.npy')
    np.save(np_file, cls_token_data)
    print(f"Saved visual features array: {cls_token_data.shape} to {np_file}")

    metadata = {
        'image_names': image_names,
        'embedding_shape': cls_token_data.shape,
        'embedding_dim': cls_token_data.shape[1],
        'num_images': len(image_names)
    }

    metadata_file = os.path.join(output_dir, 'visual_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Features saved to {output_dir}")


def main():
    print("Loading DINOv2 model...")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    print("Loading images...")
    images, image_names = load_and_process_images('data/images/')

    if not images:
        print("No images found in data/ directory")
        return

    print(f"Extracting features from {len(images)} images...")

    features = extract_features(images, model, processor)

    print("\nFeature shapes:")
    for key, value in features.items():
        print(f"{key}: {value.shape}")

    # Save features
    print("\nSaving features...")
    save_features(features, image_names)

    print("Feature extraction complete!")


if __name__ == "__main__":
    main()
