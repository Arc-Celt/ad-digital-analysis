"""
Visual annotation feature clustering with UMAP and HDBSCAN.
Saves clustering results for visualization.
"""

import numpy as np
import json
import pandas as pd
import umap
import hdbscan
from pathlib import Path
from collections import Counter


def load_annotation_features(jsonl_path):
    """Load visual annotation features from JSONL file."""
    features = []
    file_names = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            features.append(data['embedding'])
            file_names.append(data['file_name'])

    return np.array(features), file_names


def load_annotation_content(file_names, annotations_dir):
    """Load original annotation content from files."""
    annotations = []
    for file_name in file_names:
        file_path = annotations_dir / file_name
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                annotation = f.read().strip()
                annotations.append(annotation)
        except Exception as e:
            annotations.append(f"Error loading: {str(e)}")
    return annotations


def save_clustering_results(
    file_names, features_2d, cluster_labels, project_root,
    output_dir='output/clustering/'
):
    """Save clustering results to CSV file with original annotation content."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load original annotation content
    annotations_dir = project_root / 'data' / 'images' / 'annotations'
    print("Loading original annotation content...")
    annotation_contents = load_annotation_content(file_names, annotations_dir)

    # Create DataFrame
    cluster_labels_list = [
        'Cluster ' + str(c) if c != -1 else 'Noise'
        for c in cluster_labels
    ]
    df = pd.DataFrame({
        'id': file_names,
        'annotation': annotation_contents,
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'cluster': cluster_labels,
        'cluster_label': cluster_labels_list
    })

    # Save to CSV
    output_file = output_path / 'visual_annotation_clustering_results.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # Analyze results
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    n_noise = np.sum(cluster_labels == -1)

    print(f"Saved clustering results to {output_file}")
    print(
        f"  Clusters: {n_clusters}, Noise: {n_noise}, "
        f"Total: {len(file_names)}"
    )

    return output_file


def main():
    """Main clustering pipeline for visual annotation features."""
    print("=" * 60)
    print("Visual Annotation Feature Clustering")
    print("=" * 60)

    # Load features
    print("\nLoading visual annotation features...")
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    features_path = (
        project_root / 'output' / 'features' /
        'visual_annotation_features.jsonl'
    )

    features, file_names = load_annotation_features(features_path)
    print(
        f"Loaded {len(features)} annotations "
        f"with {features.shape[1]}D features"
    )

    # UMAP dimensionality reduction
    print("\nRunning UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=10,
        min_dist=0.02,
        metric='cosine',
        spread=1.0
    )
    features_2d = reducer.fit_transform(features)
    print(f"Reduced from {features.shape[1]}D to {features_2d.shape[1]}D")

    # HDBSCAN clustering
    print("\nRunning HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=6,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='leaf',
        cluster_selection_epsilon=0.1
    )
    cluster_labels = clusterer.fit_predict(features_2d)

    # Analyze results
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    n_noise = np.sum(cluster_labels == -1)

    print("\nHDBSCAN Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    print(f"  Total annotations: {len(file_names)}")

    cluster_sizes = Counter(cluster_labels)
    print("\nCluster sizes:")
    for cluster_id, size in sorted(cluster_sizes.items()):
        if cluster_id != -1:
            print(f"  Cluster {cluster_id}: {size} annotations")
        else:
            print(f"  Noise: {size} annotations")

    # Save results
    print("\nSaving clustering results...")
    output_dir = project_root / 'output' / 'clustering'
    save_clustering_results(
        file_names, features_2d, cluster_labels, project_root, output_dir
    )

    print("\nVisual annotation clustering complete!")


if __name__ == '__main__':
    main()
