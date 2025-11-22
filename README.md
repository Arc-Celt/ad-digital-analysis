# Advertisement Digital Analysis

This project analyzes historical Chinese advertisement images by extracting textual features from ad copy and visual annotations, then performing clustering analysis to discover patterns and groupings.

## Overview

The project processes advertisement images in two parallel pipelines:

1. **Textual Analysis**: Extracts embeddings from advertisement text content and clusters similar ads
2. **Visual Annotation Analysis**: Uses AI to annotate images, then clusters the annotations to discover visual patterns

## Project Structure

Please make sure to put the files at the corresponding directories before running any scripts.

```
ad-digital-analysis/
├── data/
│   ├── images/              # Source advertisement images
│   │   └── annotations/     # AI-generated image annotations
│   ├── text/                # Extracted text content from ads
│   └── all.txt              # Original consolidated text file
├── src/                     # Scripts directory
├── output/                  # Generated outputs
│   ├── features/            # Extracted embeddings
│   └── clustering/          # Clustering results and visualizations
└── models/                  # Embedding models (Qwen3-Embedding-0.6B)
```

## Workflow

### 1. Data Preparation

- **Text Extraction**: Text content is extracted from advertisement images and saved to `data/text/`
- **Deduplication**: `deduplicate_text_files.py` removes duplicate or highly similar text files (95% similarity threshold).

### 2. Feature Extraction

#### Textual Features

- **`textual_feature_extraction.py`**: Extracts embeddings from text files using Qwen3-Embedding-0.6B model
- Output: `output/features/textual_features.jsonl`

#### Visual Annotation Features

- **`annotate_images.py`**: Uses Google Gemini 2.5 Flash API to generate text annotations for each image
- **`visual_annotation_feature_extraction.py`**: Extracts embeddings from image annotations using Qwen3-Embedding-0.6B
- Output: `output/features/visual_annotation_features.jsonl`

### 3. Clustering

#### Textual Clustering

- **`textual_clustering.py`**: Performs UMAP dimensionality reduction and HDBSCAN clustering on textual embeddings
- Output: `output/clustering/textual_clustering_results.csv`

#### Visual Annotation Clustering

- **`visual_annotation_clustering.py`**: Performs UMAP dimensionality reduction and HDBSCAN clustering on annotation embeddings
- Output: `output/clustering/visual_annotation_clustering_results.csv`

### 4. Visualization

- **`visualize_textual_clustering.py`**: Creates 2D scatter plots of textual clustering results
- **`visualize_visual_annotation_clustering.py`**: Creates 2D scatter plots of visual annotation clustering results
- Outputs: PNG visualization files in `output/clustering/`

### 5. Sample Extraction

- **`save_textual_cluster_examples.py`**: Extracts 10 sample examples from each textual cluster
- **`save_visual_cluster_examples.py`**: Creates visual grid showcasing sample images from each visual cluster
- Outputs: CSV files and visualization grids in `output/clustering/samples/`

## Key Scripts

| Script                                      | Purpose                                         |
| ------------------------------------------- | ----------------------------------------------- |
| `annotate_images.py`                        | Annotate images using Gemini API                |
| `textual_feature_extraction.py`             | Extract embeddings from text files              |
| `visual_annotation_feature_extraction.py`   | Extract embeddings from image annotations       |
| `deduplicate_text_files.py`                 | Remove duplicate text files (95% similarity)    |
| `textual_clustering.py`                     | Cluster textual features using UMAP + HDBSCAN   |
| `visual_annotation_clustering.py`           | Cluster visual annotations using UMAP + HDBSCAN |
| `visualize_textual_clustering.py`           | Visualize textual clustering results            |
| `visualize_visual_annotation_clustering.py` | Visualize visual annotation clustering          |
| `save_textual_cluster_examples.py`          | Extract sample examples from textual clusters   |
| `save_visual_cluster_examples.py`           | Create visual showcase of cluster examples      |

## Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google AI Studio API key**:

   - Create a `.env` file in the project root
   - Add: `GOOGLE_AI_STUDIO_API_KEY=your_api_key_here`

3. **Download embedding model**:
   - Place Qwen3-Embedding-0.6B model in `models/Qwen3-Embedding-0.6B/`

## Usage

Run scripts in order:

```bash
# 1. Annotate images
python src/annotate_images.py

# 2. Extract textual features
python src/textual_feature_extraction.py

# 3. Extract visual annotation features
python src/visual_annotation_feature_extraction.py

# 4. Deduplicate text files (optional)
python src/deduplicate_text_files.py

# 5. Perform clustering
python src/textual_clustering.py
python src/visual_annotation_clustering.py

# 6. Visualize results
python src/visualize_textual_clustering.py
python src/visualize_visual_annotation_clustering.py

# 7. Extract samples
python src/save_textual_cluster_examples.py
python src/save_visual_cluster_examples.py
```

## Technologies

- **Embedding Model**: Qwen3-Embedding-0.6B
- **Image Annotation**: Google Gemini 2.5 Flash API
- **Dimensionality Reduction**: UMAP
- **Clustering**: HDBSCAN
- **Visualization**: Matplotlib

## Output Files

- **Features**: JSONL files with embeddings (`output/features/`)
- **Clustering Results**: CSV files with cluster assignments (`output/clustering/`)
- **Visualizations**: PNG scatter plots (`output/clustering/`)
- **Samples**: CSV files with cluster examples (`output/clustering/samples/`)
