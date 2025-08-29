import os
import cv2
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = "/imgarc/nila/data/Super_Res/all_data/full_fov_and_wbc_patch_iter_3/organised_data/train/sr_1"
TARGET_DIR = "/imgarc/nila/data/Super_Res/all_data/full_fov_and_wbc_patch_iter_3/organised_data/train/clean"
OUTPUT_DIR = "/imgarc/nila/data/Deblur_Defocus/wbc_and_fov_patch_data/train"
# -------------------

def process_image_pair(source_path):
    """Reads source and target, stacks them, and saves the result."""
    try:
        basename = os.path.basename(source_path)
        target_path = os.path.join(TARGET_DIR, basename)
        output_path = os.path.join(OUTPUT_DIR, basename)

        if not os.path.exists(target_path):
            return f"Skipped: Target for {basename} not found."

        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)

        # Ensure images have the same height for horizontal stacking
        if source_img.shape[0] != target_img.shape[0]:
            h = source_img.shape[0]
            w = int(target_img.shape[1] * (h / target_img.shape[0]))
            target_img = cv2.resize(target_img, (w, h), interpolation=cv2.INTER_AREA)

        concatenated_img = np.hstack([source_img, target_img])
        cv2.imwrite(output_path, concatenated_img)
        return None
    except Exception as e:
        return f"Failed {os.path.basename(source_path)}: {e}"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all image files in the source directory
    source_paths = glob.glob(os.path.join(SOURCE_DIR, '*.png')) + \
                   glob.glob(os.path.join(SOURCE_DIR, '*.jpg')) + \
                   glob.glob(os.path.join(SOURCE_DIR, '*.jpeg'))

    if not source_paths:
        print(f"No images found in {SOURCE_DIR}")
    else:
        print(f"Found {len(source_paths)} images to process.")
        # with ProcessPoolExecutor() as executor:
        #     # Process images in parallel and show a progress bar
        #     results = list(tqdm(executor.map(process_image_pair, source_paths), total=len(source_paths)))
        
        # # Print any errors that occurred
        # for res in results:
        #     if res:
        #         print(res)
        
        for source_path in tqdm(source_paths):
            print(source_path)
            result = process_image_pair(source_path)
            if result:
                print(result)
                
        print("\nProcessing complete.")