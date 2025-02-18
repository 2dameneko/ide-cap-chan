import os
import shutil

def copy_image_processing_file():
    # Get current Windows username
    username = os.getenv('USERNAME')
    
    # Construct paths
    source_file = os.path.join('MiniCPM-o-2_6', 'image_processing_minicpmv.py')
    destination_dir = os.path.join(
        'C:\\', 
        'Documents and Settings', 
        username, 
        '.cache', 
        'huggingface', 
        'modules', 
        'transformers_modules', 
        'MiniCPM-o-2_6'
    )
    
    # Check if source file exists
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file '{source_file}' not found")
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Perform file copy
    shutil.copy(source_file, destination_dir)
    print(f"File successfully copied to: {destination_dir}")

if __name__ == "__main__":
    copy_image_processing_file()