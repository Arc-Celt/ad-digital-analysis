"""
Visualize textual clustering results.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def setup_chinese_font():
    """Set up matplotlib to display Chinese characters."""
    plt.rcParams['font.sans-serif'] = [
        'SimHei',
        'Microsoft YaHei',
        'STHeiti',
        'Arial Unicode MS',
    ]
    plt.rcParams['axes.unicode_minus'] = False


def visualize_clustering(
    csv_path, output_path=None, max_labels_per_cluster=3
):
    """Visualize clustering results."""
    print(f"Loading clustering results from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    setup_chinese_font()
    df['display_name'] = df['id'].str.replace('.txt', '', regex=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    n_clusters = len(unique_clusters)
    n_noise = len(df[df['cluster'] == -1])

    cmap = plt.cm.get_cmap('tab20', max(20, n_clusters))
    colors = [cmap(i) for i in range(n_clusters)]
    noise_color = 'lightgray'

    print(f"Plotting {n_clusters} clusters and {n_noise} noise points...")

    for i, cluster_id in enumerate(unique_clusters):
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)

        ax.scatter(
            cluster_df['x'],
            cluster_df['y'],
            c=[colors[i]],
            alpha=0.7,
            s=80,
            label=f'Cluster {cluster_id}',
            edgecolors='black',
            linewidths=0.5
        )

        sample_size = min(max_labels_per_cluster, cluster_size)
        if sample_size > 0:
            sample_df = cluster_df.sample(n=sample_size, random_state=42)
            for idx, row in sample_df.iterrows():
                display_name = str(row['display_name'])
                ax.annotate(
                    display_name,
                    xy=(row['x'], row['y']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=12,
                    alpha=0.8,
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        alpha=0.8,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                )

    noise_df = df[df['cluster'] == -1]
    if len(noise_df) > 0:
        ax.scatter(
            noise_df['x'],
            noise_df['y'],
            c=noise_color,
            alpha=0.5,
            s=50,
            label='Noise',
            edgecolors='gray',
            linewidths=0.5,
            zorder=0
        )

    ax.set_xlabel('UMAP Dimension 1', fontsize=18)
    ax.set_ylabel('UMAP Dimension 2', fontsize=18)
    ax.set_title(
        'Textual Clustering',
        fontsize=26,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=16)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    csv_path = (
        project_root / 'output' / 'clustering' /
        'textual_clustering_results.csv'
    )

    if not csv_path.exists():
        print(f"ERROR: Clustering results file not found: {csv_path}")
        return

    output_path = (
        project_root / 'output' / 'clustering' /
        'textual_clustering_plot.png'
    )

    print("=" * 60)
    print("Textual Clustering Visualization")
    print("=" * 60)
    print()

    visualize_clustering(csv_path, output_path, max_labels_per_cluster=3)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
