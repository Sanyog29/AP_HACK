import os
import yaml
import shutil
from ultralytics import YOLO

# Define paths
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
MODEL_DIR = os.path.join(os.getcwd(), 'model')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def create_dataset_yaml():
    """
    Create a YAML configuration file for the dataset.
    This is required by YOLOv8 for training.
    """
    # Analyze the dataset to determine classes
    class_ids = set()
    
    # Check train labels to identify all class IDs
    train_labels_dir = os.path.join(DATASET_DIR, 'train', 'labels')
    if os.path.exists(train_labels_dir):
        for label_file in os.listdir(train_labels_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(train_labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_ids.add(int(parts[0]))
    
    # Create class mapping (assuming class IDs are 0, 1, etc.)
    names = {}
    for class_id in sorted(class_ids):
        if class_id == 0:
            names[class_id] = 'gunny_bag'  # Assuming class 0 is gunny bag
        else:
            names[class_id] = f'class_{class_id}'  # Generic name for other classes
    
    # If no classes found, add a default
    if not names:
        names[0] = 'gunny_bag'
    
    # Create dataset config
    dataset_config = {
        'path': DATASET_DIR,
        'train': 'train/images',
        'val': 'valid/images',  # Using 'valid' folder as validation
        'test': 'test/images',
        'names': names
    }
    
    # Write the YAML file
    yaml_path = os.path.join(DATASET_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset configuration at {yaml_path}")
    print(f"Detected classes: {names}")
    
    return yaml_path

def train_model():
    """
    Train a YOLOv8 model on the dataset.
    """
    # Create dataset YAML
    yaml_path = create_dataset_yaml()
    
    # Initialize model (start with a pre-trained YOLOv8 model)
    model = YOLO('yolov8n.pt')  # Using the smallest model for faster training
    
    # Train the model
    print("Starting model training...")
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,  # Early stopping
        save=True,
        name='gunny_bag_detector'
    )
    
    # Copy the best model to our model directory
    best_model_path = os.path.join(os.getcwd(), 'runs', 'detect', 'gunny_bag_detector', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, os.path.join(MODEL_DIR, 'best.pt'))
        print(f"Best model saved to {os.path.join(MODEL_DIR, 'best.pt')}")
    else:
        print("Training completed but best model not found at expected path")
        # Try to find the model in a different location
        runs_dir = os.path.join(os.getcwd(), 'runs')
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                if file == 'best.pt':
                    src_path = os.path.join(root, file)
                    shutil.copy(src_path, os.path.join(MODEL_DIR, 'best.pt'))
                    print(f"Found and copied best model from {src_path}")
                    break

def validate_dataset_structure():
    """
    Validate that the dataset has the expected structure.
    """
    required_dirs = [
        os.path.join(DATASET_DIR, 'train', 'images'),
        os.path.join(DATASET_DIR, 'train', 'labels'),
        os.path.join(DATASET_DIR, 'valid', 'images'),
        os.path.join(DATASET_DIR, 'valid', 'labels')
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Warning: Expected directory {dir_path} not found")
            return False
    
    # Check if there are images in the train directory
    train_images = os.path.join(DATASET_DIR, 'train', 'images')
    if not os.listdir(train_images):
        print(f"Error: No images found in {train_images}")
        return False
    
    return True

if __name__ == "__main__":
    print("Gunny Bag Detector - Training Script")
    print("====================================")
    
    # Check if dataset structure is valid
    if not validate_dataset_structure():
        print("Dataset structure is incomplete. Please organize your dataset as follows:")
        print("dataset/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  ├── valid/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  └── test/")
        print("      ├── images/")
        print("      └── labels/")
        
        print("\nYou can use prepare_data.py to organize your dataset.")
        exit(1)
    
    
    train_model()
    
    print("\nTraining complete! The best model has been saved to the 'model' folder.")
    print("You can now use this model in your Flask application.")