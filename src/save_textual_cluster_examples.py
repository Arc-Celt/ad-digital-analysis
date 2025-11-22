"""
Extract sample examples from each textual cluster and save to CSV.
"""

import pandas as pd
from pathlib import Path


def extract_product_name(filename):
    """Extract product name from filename."""
    base_name = filename.replace('.txt', '')
    parts = base_name.split()
    if parts:
        return parts[-1]
    return filename


def normalize_text(text):
    """Normalize multi-line text to single line with space separators."""
    if pd.isna(text):
        return ""
    # Split by newlines, strip whitespace, filter empty lines
    lines = [line.strip() for line in str(text).split('\n') if line.strip()]
    # Join with spaces
    return ' '.join(lines)


def extract_cluster_samples(csv_path, output_dir, samples_per_cluster=10):
    """Extract sample examples from each cluster."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading clustering results from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    df['product_name'] = df['id'].apply(extract_product_name)
    df['text_normalized'] = df['text'].apply(normalize_text)

    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    n_clusters = len(unique_clusters)

    print(f"Found {n_clusters} clusters. Extracting samples...")

    samples_data = []

    for cluster_id in unique_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)

        sample_size = min(samples_per_cluster, cluster_size)
        sample_df = cluster_df.sample(n=sample_size, random_state=42)

        print(
            f"Cluster {cluster_id}: {cluster_size} files, "
            f"sampling {sample_size}"
        )

        for _, row in sample_df.iterrows():
            samples_data.append({
                'cluster': cluster_id,
                'cluster_label': f'Cluster {cluster_id}',
                'filename': row['id'],
                'product_name': row['product_name'],
                'text': row['text_normalized']
            })

    samples_df = pd.DataFrame(samples_data)
    output_csv = output_path / 'textual_cluster_samples.csv'
    samples_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\nSamples CSV saved to: {output_csv}")
    print(f"Total samples: {len(samples_data)}")
    print(f"Number of clusters: {len(unique_clusters)}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    csv_path = (
        project_root / 'output' / 'clustering' /
        'textual_clustering_results.csv'
    )
    output_dir = project_root / 'output' / 'clustering' / 'samples'

    if not csv_path.exists():
        print(f"ERROR: Clustering results file not found: {csv_path}")
        return

    extract_cluster_samples(csv_path, output_dir, samples_per_cluster=10)
    print("Textual cluster samples extraction complete!")


if __name__ == '__main__':
    main()
