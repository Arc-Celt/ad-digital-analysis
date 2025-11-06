"""
Split the all.txt file into separate product files.
"""

from pathlib import Path


def split_products(input_file, output_dir):
    """Split the input file into separate product files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_product = []
    product_count = 0

    for line in lines:
        line = line.rstrip('\n\r')

        if not line.strip():
            if current_product:
                save_product(current_product, output_path)
                product_count += 1
                current_product = []
        else:
            current_product.append(line)

    if current_product:
        save_product(current_product, output_path)
        product_count += 1

    print(f"Successfully split {product_count} products into separate files.")
    print(f"Output directory: {output_path}")


def save_product(product_lines, output_dir):
    """Save a product's content to a file."""
    if not product_lines:
        return

    header = product_lines[0].strip()
    filename = header + '.txt'
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        for line in product_lines[1:]:
            f.write(line + '\n')


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / 'data' / 'all.txt'
    output_dir = project_root / 'data' / 'text'

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        exit(1)

    split_products(input_file, output_dir)
