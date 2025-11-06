"""
Visual feature clustering with UMAP and HDBSCAN.
Saves clustering results for visualization.
"""

import numpy as np
import json
import pandas as pd
import umap
import hdbscan
from pathlib import Path
from collections import Counter


def load_visual_features(jsonl_path):
    """Load visual features from JSONL file."""
    features = []
    image_names = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            features.append(data['embedding'])
            image_names.append(data['image_name'])
    
    return np.array(features), image_names


def save_clustering_results(
    image_names, features_2d, cluster_labels, project_root,
    output_dir='output/clustering/'
):
    """Save clustering results to CSV file with image paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create relative image paths
    image_paths = [f"data/images/{name}" for name in image_names]

    # Create DataFrame
    cluster_labels_list = [
        'Cluster ' + str(c) if c != -1 else 'Noise'
        for c in cluster_labels
    ]
    df = pd.DataFrame({
        'id': image_names,
        'image_path': image_paths,
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'cluster': cluster_labels,
        'cluster_label': cluster_labels_list
    })
    
    # Save to CSV
    output_file = output_path / 'visual_clustering_results.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Analyze results
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"Saved clustering results to {output_file}")
    print(f"  Clusters: {n_clusters}, Noise: {n_noise}, Total: {len(image_names)}")
    
    return output_file


def main():
    """Main clustering pipeline for visual features."""
    print("=" * 60)
    print("Visual Feature Clustering")
    print("=" * 60)
    
    # Load features
    print("\nLoading visual features...")
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    features_path = project_root / 'output' / 'features' / 'visual_features.jsonl'
    
    features, image_names = load_visual_features(features_path)
    print(f"Loaded {len(features)} images with {features.shape[1]}D features")
    
    # UMAP dimensionality reduction
    print("\nRunning UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=10,
        min_dist=0.1,
        metric='euclidean'
    )
    features_2d = reducer.fit_transform(features)
    print(f"Reduced from {features.shape[1]}D to {features_2d.shape[1]}D")
    
    # HDBSCAN clustering
    print("\nRunning HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.1,
        algorithm='best'
    )
    cluster_labels = clusterer.fit_predict(features_2d)
    
    # Analyze results
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    n_noise = np.sum(cluster_labels == -1)
    
    print("\nHDBSCAN Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    print(f"  Total images: {len(image_names)}")
    
    cluster_sizes = Counter(cluster_labels)
    print(f"\nCluster sizes:")
    for cluster_id, size in sorted(cluster_sizes.items()):
        if cluster_id != -1:
            print(f"  Cluster {cluster_id}: {size} images")
        else:
            print(f"  Noise: {size} images")
    
    # Save results
    print("\nSaving clustering results...")
    output_dir = project_root / 'output' / 'clustering'
    save_clustering_results(
        image_names, features_2d, cluster_labels, project_root, output_dir
    )
    
    print("\nVisual clustering complete!")


if __name__ == '__main__':
    main()
