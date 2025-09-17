import os
import shutil
import glob
import random
from pathlib import Path

# --- Configuration ---
MERGED_FOLDER = "/imgarc/nila/data/Deblur_Defocus/wbc_and_fov_patch_data/merged"  # Update this path
OUTPUT_BASE_DIR = "/imgarc/nila/data/Deblur_Defocus/wbc_and_fov_patch_data"     # Update this path

# Split ratios (should sum to 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

# File extensions to include
FILE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp']
# -------------------

def get_all_files(folder_path, extensions):
    """Get all files with specified extensions from the folder."""
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return all_files

def create_directories(base_dir):
    """Create train, val, and test directories."""
    dirs = {
        'train': os.path.join(base_dir, 'train'),
        'val': os.path.join(base_dir, 'val'),
        'test': os.path.join(base_dir, 'test')
    }
    
    for split_name, dir_path in dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return dirs

def split_files(file_list, train_ratio, val_ratio, test_ratio):
    """Split file list into train, val, and test sets."""
    total_files = len(file_list)
    
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count  # Remaining files go to test
    
    # Shuffle files for random distribution
    random.shuffle(file_list)
    
    splits = {
        'train': file_list[:train_count],
        'val': file_list[train_count:train_count + val_count],
        'test': file_list[train_count + val_count:]
    }
    
    print(f"Split distribution:")
    print(f"  Train: {len(splits['train'])} files ({len(splits['train'])/total_files*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} files ({len(splits['val'])/total_files*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} files ({len(splits['test'])/total_files*100:.1f}%)")
    
    return splits

def move_files(splits, target_dirs):
    """Move files to their respective directories."""
    moved_count = {'train': 0, 'val': 0, 'test': 0}
    failed_moves = []
    
    for split_name, file_list in splits.items():
        target_dir = target_dirs[split_name]
        
        print(f"\nMoving files to {split_name} directory...")
        for file_path in file_list:
            try:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_dir, filename)
                
                # Check if file already exists in target
                if os.path.exists(target_path):
                    print(f"Warning: {filename} already exists in {split_name}, skipping...")
                    continue
                
                shutil.move(file_path, target_path)
                moved_count[split_name] += 1
                
            except Exception as e:
                error_msg = f"Failed to move {os.path.basename(file_path)} to {split_name}: {e}"
                failed_moves.append(error_msg)
                print(error_msg)
    
    return moved_count, failed_moves

def main():
    # Set random seed
    random.seed(RANDOM_SEED)
    
    # Validate input directory
    if not os.path.exists(MERGED_FOLDER):
        print(f"Error: Merged folder '{MERGED_FOLDER}' does not exist!")
        return
    
    # Validate split ratios
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Split ratios don't sum to 1.0 (current sum: {total_ratio})")
        return
    
    print(f"Starting dataset split...")
    print(f"Source directory: {MERGED_FOLDER}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Split ratios - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")
    
    # Get all files
    all_files = get_all_files(MERGED_FOLDER, FILE_EXTENSIONS)
    
    if not all_files:
        print(f"No files found in {MERGED_FOLDER} with extensions {FILE_EXTENSIONS}")
        return
    
    print(f"Found {len(all_files)} files to split")
    
    # Create output directories
    target_dirs = create_directories(OUTPUT_BASE_DIR)
    
    # Split files
    splits = split_files(all_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Move files
    moved_count, failed_moves = move_files(splits, target_dirs)
    
    # Summary
    print("\n" + "="*50)
    print("SPLIT COMPLETE!")
    print("="*50)
    print(f"Successfully moved:")
    for split_name, count in moved_count.items():
        print(f"  {split_name}: {count} files")
    
    if failed_moves:
        print(f"\nFailed moves: {len(failed_moves)}")
        for error in failed_moves:
            print(f"  {error}")
    
    # Check if source directory is empty
    remaining_files = get_all_files(MERGED_FOLDER, FILE_EXTENSIONS)
    if remaining_files:
        print(f"\nWarning: {len(remaining_files)} files remain in source directory")
    else:
        print(f"\nSource directory is now empty (all files moved)")

if __name__ == "__main__":
    main()