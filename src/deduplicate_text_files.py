"""
Remove duplicate or highly similar text files.
Only keeps one file from each group of similar files.
"""

from pathlib import Path
from difflib import SequenceMatcher


def read_text_file(file_path):
    """Read and normalize text content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Normalize whitespace
            content = ' '.join(content.split())
            return content
    except Exception as e:
        return f"Error reading file: {e}"


def similarity(text1, text2):
    """Calculate similarity ratio between two texts."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


def deduplicate_text_files(text_dir, similarity_threshold=0.95):
    """Remove duplicate or highly similar text files."""
    text_path = Path(text_dir)

    if not text_path.exists():
        print(f"Error: Directory not found: {text_dir}")
        return

    # Read all text files with their full paths
    files_content = {}
    for file_path in sorted(text_path.glob('*.txt')):
        content = read_text_file(file_path)
        if content and not content.startswith("Error"):
            files_content[file_path] = content

    num_files = len(files_content)
    print(f"Checking {num_files} text files for duplicates...\n")

    # Find duplicate groups
    file_list = list(files_content.items())
    duplicate_groups = []
    processed = set()

    for i, (file1, content1) in enumerate(file_list):
        if file1 in processed:
            continue

        similar_files = [file1]

        for file2, content2 in file_list[i+1:]:
            if file2 in processed:
                continue

            sim = similarity(content1, content2)
            if sim >= similarity_threshold:
                similar_files.append(file2)
                processed.add(file2)

        if len(similar_files) > 1:
            duplicate_groups.append(similar_files)
            processed.add(file1)

    if not duplicate_groups:
        threshold_pct = similarity_threshold * 100
        msg = f"No duplicate files found (similarity >= {threshold_pct:.0f}%)."
        print(msg)
        return

    threshold_pct = similarity_threshold * 100
    print("=" * 60)
    print(f"DUPLICATE GROUPS FOUND (similarity >= {threshold_pct:.0f}%):")
    print("=" * 60)

    total_to_delete = 0

    for i, group in enumerate(duplicate_groups, 1):
        print(f"\nGroup {i} ({len(group)} files):")
        # Keep the first file, mark others for deletion
        keep_file = group[0]
        delete_files = group[1:]

        print(f"  KEEP: {keep_file.name}")
        for delete_file in delete_files:
            print(f"  DELETE: {delete_file.name}")

        total_to_delete += len(delete_files)

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print(f"Total duplicate groups: {len(duplicate_groups)}")
    print(f"Files to keep: {len(duplicate_groups)}")
    print(f"Files to delete: {total_to_delete}")

    # Confirm deletion
    if total_to_delete > 0:
        print(f"\nProceeding to delete {total_to_delete} duplicate files...")
        deleted_count = 0

        for group in duplicate_groups:
            keep_file = group[0]
            delete_files = group[1:]

            for delete_file in delete_files:
                try:
                    delete_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    msg = f"  Error deleting {delete_file.name}: {e}"
                    print(msg)

        print(f"\nSuccessfully deleted {deleted_count} duplicate files.")
        remaining = len(files_content) - deleted_count
        print(f"Remaining files: {remaining}")
    else:
        print("\nNo files to delete.")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    text_dir = project_root / 'data' / 'text'
    deduplicate_text_files(text_dir, similarity_threshold=0.95)


if __name__ == '__main__':
    main()
