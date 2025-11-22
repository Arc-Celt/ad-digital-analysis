"""
Showcase examples from each cluster.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image


def extract_product_name(filename):
    """Extract product name."""
    base_name = filename.replace('.txt', '')
    parts = base_name.split()
    if parts:
        return parts[-1]
    return filename


def setup_chinese_font():
    """Set up matplotlib to display Chinese characters."""
    plt.rcParams['font.sans-serif'] = [
        'SimHei',
        'Microsoft YaHei',
        'STHeiti',
        'Arial Unicode MS',
    ]
    plt.rcParams['axes.unicode_minus'] = False


def showcase_clusters(csv_path, images_dir, output_dir):
    """Create visual showcase of cluster examples."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['product_name'] = df['id'].apply(extract_product_name)
    df['image_filename'] = df['id'].str.replace('.txt', '.png')

    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    n_clusters = len(unique_clusters)

    examples_data = []

    n_cols = 5
    fig = plt.figure(figsize=(n_cols * 2.5, n_clusters * 3))

    height_ratios = []
    for _ in range(n_clusters):
        height_ratios.append(0.3)
        height_ratios.append(1.0)

    gs = GridSpec(
        n_clusters * 2, n_cols, figure=fig,
        hspace=0.15, wspace=0.05,
        height_ratios=height_ratios,
        top=0.97, bottom=0.05
    )

    for cluster_idx, cluster_id in enumerate(unique_clusters):
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)

        sample_size = min(5, cluster_size)
        sample_df = cluster_df.sample(n=sample_size, random_state=42)

        title_row = cluster_idx * 2
        title_ax = fig.add_subplot(gs[title_row, :])
        title_ax.axis('off')
        title_ax.text(
            0.5, 0.5,
            f'Cluster {cluster_id} (Count: {cluster_size})',
            ha='center',
            va='center',
            fontsize=20,
            fontweight='bold'
        )

        images_row = cluster_idx * 2 + 1
        for example_idx, (_, row) in enumerate(sample_df.iterrows()):
            ax = fig.add_subplot(gs[images_row, example_idx])
            ax.axis('off')

            image_path = images_dir / row['image_filename']
            if image_path.exists():
                try:
                    img = Image.open(image_path)
                    ax.imshow(img)
                    ax.set_title(
                        row['product_name'],
                        fontsize=14,
                        fontweight='bold',
                        pad=2
                    )
                except Exception:
                    ax.text(
                        0.5, 0.5,
                        f'Error loading\n{row["image_filename"]}',
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='red'
                    )
            else:
                ax.text(
                    0.5, 0.5,
                    f'Image not found\n{row["image_filename"]}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='red'
                )

            examples_data.append({
                'cluster': cluster_id,
                'cluster_label': f'Cluster {cluster_id}',
                'filename': row['id'],
                'image_filename': row['image_filename'],
                'product_name': row['product_name'],
                'annotation': row['annotation']
            })

    plt.suptitle(
        'Visual Annotation Clustering Examples',
        fontsize=28,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_plot = output_path / 'visual_annotation_clustering_examples.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_plot}")

    examples_df = pd.DataFrame(examples_data)
    output_csv = output_path / 'visual_annotation_clustering_examples.csv'
    examples_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Examples CSV saved to: {output_csv}")

    print(f"Showcase complete! {len(examples_data)} examples processed.")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    csv_path = (
        project_root / 'output' / 'clustering' /
        'visual_annotation_clustering_results.csv'
    )
    images_dir = project_root / 'data' / 'images'
    output_dir = project_root / 'output' / 'clustering' / 'samples'

    if not csv_path.exists():
        print(f"ERROR: Clustering results file not found: {csv_path}")
        return

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return

    setup_chinese_font()

    showcase_clusters(csv_path, images_dir, output_dir)
    print("Cluster examples showcase complete!")


if __name__ == '__main__':
    main()
