import os
import sys

# --- 1. CONFIGURE THESE ---
# Put the full, absolute path to your .jpg images here
# Added 'r' to fix Windows path errors
IMAGE_DIR = r"C:\Users\Sabar\OneDrive\Documents\DeepCrack-master\image"  

# Put the full, absolute path to your .bmp masks here
# Added 'r' to fix Windows path errors
MASK_DIR = r"C:\Users\Sabar\OneDrive\Documents\DeepCrack-master\gt"    

# This is the output file the script will create.
# The main 'Config' class will read this file.
OUTPUT_FILE = r"data\my_val_data.txt"  # Also made this a raw string, just in case
# --------------------------


def create_manifest():
    """
    Generates a text file mapping absolute image paths to absolute mask paths.
    
    Assumes that for every image 'foo.jpg' in IMAGE_DIR, there is a
    corresponding mask 'foo.bmp' in MASK_DIR.
    """
    print(f"Starting manifest creation...")
    print(f"Image Directory: {IMAGE_DIR}")
    print(f"Mask Directory: {MASK_DIR}")
    print(f"Output File: {OUTPUT_FILE}")

    # Check if directories exist
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Image directory not found at {IMAGE_DIR}")
        print("Please update the IMAGE_DIR variable in this script.")
        sys.exit(1)
        
    if not os.path.isdir(MASK_DIR):
        print(f"Error: Mask directory not found at {MASK_DIR}")
        print("Please update the MASK_DIR variable in this script.")
        sys.exit(1)

    file_pairs = []
    
    # Get a sorted list of all image files
    try:
        image_files = sorted(os.listdir(IMAGE_DIR))
    except FileNotFoundError:
        print(f"Error: Could not list files in {IMAGE_DIR}.")
        return

    if not image_files:
        print(f"Warning: No images found in {IMAGE_DIR}.")
        return

    print(f"Found {len(image_files)} potential images. Checking for masks...")

    # Loop through all images found
    for img_name in image_files:
        # Skip hidden files like .DS_Store
        if img_name.startswith('.'):
            continue

        # Get the base name without extension (e.g., '6192')
        base_name, img_ext = os.path.splitext(img_name)
        
        # Only process files with expected extensions (like .jpg, .jpeg)
        if img_ext.lower() not in ['.jpg', '.jpeg', '.png']:
            print(f"Skipping non-image file: {img_name}")
            continue

        # --- This is the key part ---
        # Look for a .bmp mask file with the same base name
        mask_name = base_name + ".bmp" 
        
        # Get absolute paths
        img_path = os.path.abspath(os.path.join(IMAGE_DIR, img_name))
        mask_path = os.path.abspath(os.path.join(MASK_DIR, mask_name))

        # Only add the pair if both files actually exist
        if os.path.exists(mask_path):
            file_pairs.append(f"{img_path} {mask_path}")
        else:
            # This warning is crucial for debugging!
            print(f"Warning: Found image {img_name}, but missing mask {mask_path}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        # Create 'data' directory if it doesn't exist
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)

    # Write all the found pairs to the output file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            for pair in file_pairs:
                f.write(pair + "\n")
    except IOError as e:
        print(f"Error writing to output file {OUTPUT_FILE}: {e}")
        return

    print("-" * 30)
    print(f"Done! Created {OUTPUT_FILE} with {len(file_pairs)} pairs.")
    if len(file_pairs) == 0:
        print("Warning: No pairs were found. Check your paths and file names.")
    print("-" * 30)


# This makes the script runnable from the command line
if _name_ == "_main_":
    create_manifest()