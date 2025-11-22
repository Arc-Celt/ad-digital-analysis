"""
Annotate images using Gemini API.
"""

import os
import sys
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai


def load_prompt(prompt_file):
    """Load the annotation prompt from file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def setup_genai(api_key):
    """Set up Google Generative AI with API key."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model


def get_api_key():
    """Get API key from .env file or environment variable."""
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")
    return os.getenv('GOOGLE_AI_STUDIO_API_KEY')


def annotate_image(model, image_path, prompt):
    """
    Annotate a single image using the Gemini model.

    Args:
        model: The Gemini model instance
        image_path: Path to the image file
        prompt: The annotation prompt

    Returns:
        The annotation text or None if failed
    """
    try:
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])

        # Extract text from response
        if response.text:
            return response.text.strip()
        else:
            print(f"  Warning: Empty response for {image_path.name}")
            return None

    except Exception as e:
        print(f"  Error annotating {image_path.name}: {str(e)}")
        return None


def main():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: GOOGLE_AI_STUDIO_API_KEY not found in .env file")
        sys.exit(1)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    images_dir = project_root / "data" / "images"
    annotations_dir = images_dir / "annotations"
    prompt_file = project_root / "data" / "image_annotation_prompt.md"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    if not prompt_file.exists():
        print(f"ERROR: Prompt file not found: {prompt_file}")
        sys.exit(1)

    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt
    print("Loading annotation prompt...")
    prompt = load_prompt(prompt_file)
    print(f"Prompt loaded ({len(prompt)} characters)\n")

    # Set up Gemini model
    print("Setting up Gemini API...")
    try:
        model = setup_genai(api_key)
        print("Gemini API configured successfully\n")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini API: {str(e)}")
        sys.exit(1)

    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    image_files = [
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images to annotate\n")

    successful = 0
    skipped = 0
    failed = 0

    for i, image_path in enumerate(image_files, 1):
        annotation_path = annotations_dir / f"{image_path.stem}.txt"

        # Skip if already annotated
        if annotation_path.exists():
            msg = (
                f"[{i}/{len(image_files)}] "
                f"Skipping {image_path.name} (already annotated)"
            )
            print(msg)
            skipped += 1
            continue

        print(f"[{i}/{len(image_files)}] Annotating {image_path.name}...")

        # Annotate image
        annotation = annotate_image(model, image_path, prompt)

        if annotation:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                f.write(annotation)
            print(f"  ✓ Saved to {annotation_path.name}")
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print("Annotation complete!")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(image_files)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
