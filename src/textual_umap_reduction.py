import numpy as np
import json
import os
import matplotlib.pyplot as plt
import umap


def load_textual_features(jsonl_path):
    """Load textual features from JSONL file"""
    features = []
    file_names = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            features.append(data['embedding'])
            file_names.append(data['file_name'])

    return np.array(features), file_names


def reduce_with_umap(features, n_components=2, n_neighbors=15, min_dist=0.1,
                     random_state=42):
    """Reduce embeddings using UMAP"""
    
    print(f"Original embeddings shape: {features.shape}")
    print(f"Reducing to {n_components}D using UMAP...")
    
    # Create UMAP reducer
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='cosine'  # Good for normalized embeddings
    )
    
    # Fit and transform
    features_reduced = reducer.fit_transform(features)
    
    print(f"Reduced embeddings shape: {features_reduced.shape}")
    
    return features_reduced, reducer

def save_reduced_features(features_reduced, file_names, output_dir='output/features/'):
    """Save reduced features to files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSONL
    jsonl_file = os.path.join(output_dir, 'textual_features_2d.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i, (file_name, embedding) in enumerate(zip(file_names, features_reduced)):
            record = {
                'file_name': file_name,
                'embedding_2d': embedding.tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(file_names)} 2D embeddings to {jsonl_file}")
    
    # Also save as numpy
    np_file = os.path.join(output_dir, 'textual_features_2d.npy')
    np.save(np_file, features_reduced)
    print(f"Saved 2D embeddings array: {features_reduced.shape} to {np_file}")
    
    # Save metadata
    metadata = {
        'file_names': file_names,
        'embedding_shape': features_reduced.shape,
        'embedding_dim': features_reduced.shape[1],
        'num_files': len(file_names),
        'umap_params': {
            'n_components': features_reduced.shape[1],
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'cosine'
        }
    }

    with open(os.path.join(output_dir, 'textual_2d_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"2D features saved to {output_dir}")

def visualize_umap(features_reduced, file_names, output_dir='output/features/'):
    """Visualize UMAP results"""
    
    plt.figure(figsize=(12, 8))
    plt.scatter(features_reduced[:, 0], features_reduced[:, 1], alpha=0.7, s=50)
    
    # Add some labels
    for i, name in enumerate(file_names[:10]):  # Label first 10
        plt.annotate(name.replace('.txt', ''), 
                    (features_reduced[i, 0], features_reduced[i, 1]), 
                    fontsize=8, alpha=0.8)
    
    plt.title('UMAP Reduction of Textual Embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'textual_umap_visualization.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"UMAP visualization saved to {plot_file}")
    
    plt.show()

def main():
    """Main function to reduce textual embeddings with UMAP"""
    
    # Load textual features
    print("Loading textual features...")
    features, file_names = load_textual_features('output/features/textual_features.jsonl')
    print(f"Loaded {len(features)} text embeddings with {features.shape[1]}D features")
    
    # Reduce with UMAP
    features_2d, reducer = reduce_with_umap(features)
    
    # Save reduced features
    print("\nSaving reduced features...")
    save_reduced_features(features_2d, file_names)
    
    # Visualize
    print("\nCreating visualization...")
    visualize_umap(features_2d, file_names)
    
    print("\nUMAP reduction complete!")
    print(f"Original: {features.shape[1]}D -> Reduced: {features_2d.shape[1]}D")

if __name__ == "__main__":
    main()
