import os
import shutil
import random
from glob import glob

def prepare_dataset(images_dir, split_ratio=0.8):
    """
    Prepare dataset by splitting images into train and validation sets.
    
    Args:
        images_dir: Directory containing images and annotations
        split_ratio: Train/validation split ratio (default: 0.8)
    """
    # Create dataset structure
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    train_label_dir = os.path.join(dataset_dir, 'labels', 'train')
    val_label_dir = os.path.join(dataset_dir, 'labels', 'val')
    
    # Create directories if they don't exist
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob(os.path.join(images_dir, f'*.{ext}')))
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    # Copy files to train and validation directories
    for img_path in train_images:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # Copy image
        shutil.copy(img_path, os.path.join(train_img_dir, filename))
        
        # Copy label if exists
        label_path = os.path.join(images_dir, f'{base_name}.txt')
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(train_label_dir, f'{base_name}.txt'))
    
    for img_path in val_images:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # Copy image
        shutil.copy(img_path, os.path.join(val_img_dir, filename))
        
        # Copy label if exists
        label_path = os.path.join(images_dir, f'{base_name}.txt')
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(val_label_dir, f'{base_name}.txt'))
    
    print(f"Dataset prepared: {len(train_images)} training images, {len(val_images)} validation images")

if __name__ == "__main__":
    # Check if source directory is provided
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prepare_data.py <path_to_images_and_labels>")
        sys.exit(1)
    
    images_dir = sys.argv[1]
    if not os.path.exists(images_dir):
        print(f"Error: Directory {images_dir} does not exist")
        sys.exit(1)
    
    prepare_dataset(images_dir)