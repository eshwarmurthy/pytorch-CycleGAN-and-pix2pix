import time
import cv2
import numpy as np
import onnxruntime as ort
from glob import glob
import os
from pathlib import Path
import gc # Import the Garbage Collector interface

# --- Helper functions (Unchanged) ---

def load_and_preprocess(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    chw = np.transpose(img, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0).astype(np.float16)
    return nchw, img_bgr

def postprocess(nchw):
    chw = nchw[0]
    chw = (chw * 0.5) + 0.5
    chw = np.clip(chw, 0.0, 1.0)
    hwc = np.transpose(chw, (1, 2, 0))
    rgb_u8 = (hwc * 255.0).round().astype(np.uint8)
    return rgb_u8

def add_text_label(img, text, position=(10, 30)):
    """Add text label to image"""
    img_labeled = img.copy()
    cv2.putText(img_labeled, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 255, 0), 2, cv2.LINE_AA) # Increased font size for visibility
    return img_labeled

def reinhard_color_transfer(source, target):
    return source
    """Performs Reinhard color transfer from a source image to a target image."""
    # This function is correct and remains unchanged.
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    (s_l, s_a, s_b) = cv2.split(source_lab)
    (t_l, t_a, t_b) = cv2.split(target_lab)
    (s_l_mean, s_l_std) = (s_l.mean(), s_l.std())
    (s_a_mean, s_a_std) = (s_a.mean(), s_a.std())
    (s_b_mean, s_b_std) = (s_b.mean(), s_b.std())
    (t_l_mean, t_l_std) = (t_l.mean(), t_l.std())
    (t_a_mean, t_a_std) = (t_a.mean(), t_a.std())
    (t_b_mean, t_b_std) = (t_b.mean(), t_b.std())
    s_l -= s_l_mean
    s_a -= s_a_mean
    s_b -= s_b_mean
    s_l = (t_l_std / (s_l_std + 1e-6)) * s_l
    s_a = (t_a_std / (s_a_std + 1e-6)) * s_a
    s_b = (t_b_std / (s_b_std + 1e-6)) * s_b
    s_l += t_l_mean
    s_a += t_a_mean
    s_b += t_b_mean
    s_l = np.clip(s_l, 0, 255)
    s_a = np.clip(s_a, 0, 255)
    s_b = np.clip(s_b, 0, 255)
    transfer_lab = cv2.merge([s_l, s_a, s_b]).astype("uint8")
    transfer_bgr = cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2BGR)
    return transfer_bgr

# --- Core Logic (Refactored for Memory Efficiency) ---

def run_inference_sequentially(model_paths, input_tensor, providers=None):
    """
    Runs inference on multiple ONNX models sequentially, loading and unloading
    each one to conserve memory.
    """
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    results, timings, model_names = [], [], []
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        model_names.append(model_name)
        sess = None
        try:
            # 1. LOAD model
            print(f"  - Loading model: {model_name}")
            sess = ort.InferenceSession(model_path, providers=providers)
            
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            
            # 2. INFER
            t0 = time.time()
            output = sess.run([output_name], {input_name: input_tensor})[0]
            dt = (time.time() - t0) * 1000
            
            results.append(output)
            timings.append(dt)
            print(f"    ✓ Inference complete: {dt:.2f}ms")
            
        except Exception as e:
            print(f"    ✗ FAILED on {model_name}: {e}")
            results.append(None)
            timings.append(float('inf'))
            
        finally:
            # 3. UNLOAD model to free memory (CRITICAL STEP)
            if sess is not None:
                del sess
                gc.collect()
                
    return results, timings, model_names


def create_comparison_image(orig_img, model_outputs, model_names, clean_img=None, target_height=512, images_per_row=6):
    """Create a flexible grid of comparison images."""
    all_images = []
    
    h, w = orig_img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    
    # Add original image
    orig_resized = cv2.resize(orig_img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    all_images.append(add_text_label(orig_resized, "Original"))
    
    # Add model outputs
    for output, name in zip(model_outputs, model_names):
        if output is not None:
            processed_img = postprocess(output)
            img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (new_w, target_height), interpolation=cv2.INTER_AREA)
            if clean_img is not None:
                img_resized = reinhard_color_transfer(img_resized, cv2.resize(clean_img, (new_w, target_height)))
            all_images.append(add_text_label(img_resized, name))
        else:
            error_img = np.zeros((target_height, new_w, 3), dtype=np.uint8)
            all_images.append(add_text_label(error_img, f"{name}-ERROR"))
    
    # Add clean/ground truth image
    if clean_img is not None:
        clean_resized = cv2.resize(clean_img, (new_w, target_height), interpolation=cv2.INTER_AREA)
        all_images.append(add_text_label(clean_resized, "Ground Truth"))
    
    # Arrange images into a grid
    rows = []
    for i in range(0, len(all_images), images_per_row):
        row_images = all_images[i:i + images_per_row]
        
        # Pad the last row if it's not full
        num_missing = images_per_row - len(row_images)
        if num_missing > 0:
            padding_img = np.zeros((target_height, new_w, 3), dtype=np.uint8)
            row_images.extend([padding_img] * num_missing)
            
        rows.append(np.hstack(row_images))

    return np.vstack(rows) if rows else np.array([])


def main():
    clean_img_dir = '/imgarc/nila/data/Super_Res/all_data/full_fov_and_wbc_patch_iter_3/all_target_images_full_fov'
    test_img_dir = '/imgarc/nila/data/Super_Res/all_data/full_fov_and_wbc_patch_iter_3/all_images'
    out_dir = '/imgarc/nila/data/Deblur_Defocus/inference_all_models_v7_to_v13_full_fov'
    model_paths = sorted(glob("/imgarc/nila/data/Deblur_Defocus/Models/fp16_models/*.onnx"))
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not model_paths:
        print("Error: No .onnx models found in the specified directory.")
        return

    try:
        test_image_paths = glob(f"{test_img_dir}/*.png")
        print(f"\n🚀 Found {len(model_paths)} models and {len(test_image_paths)} images to process.\n")
        
        for i, image_path in enumerate(test_image_paths):
            print(f"--- Processing image {i+1}/{len(test_image_paths)}: {os.path.basename(image_path)} ---")
            
            clean_img_path = f"{clean_img_dir}/{os.path.basename(image_path)}"
            clean_img = cv2.imread(clean_img_path, cv2.IMREAD_COLOR) if os.path.exists(clean_img_path) else None
            
            input_tensor, orig_bgr = load_and_preprocess(image_path)
            
            # Run inference sequentially to save memory
            model_outputs, timings, model_names = run_inference_sequentially(model_paths, input_tensor)
            
            # Create comparison image
            comparison_img = create_comparison_image(
                orig_bgr, model_outputs, model_names, clean_img, 
                target_height=512, # Set a fixed height for consistency
                images_per_row=6  # Adjust this number as needed
            )
            
            # Save comparison image
            save_path = os.path.join(out_dir, f"comparison_{os.path.basename(image_path)}")
            cv2.imwrite(save_path, comparison_img)
            print(f"✓ Saved comparison to: {save_path}\n")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()