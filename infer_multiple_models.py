import time
import cv2
import numpy as np
import onnxruntime as ort
from glob import glob
import os
from pathlib import Path


def load_and_preprocess(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    chw = np.transpose(img, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0).astype(np.float32)
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
    cv2.putText(img_labeled, text, position, cv2.FONT_HERSHEY_COMPLEX, 
                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img_labeled


class MultiModelInference:
    def __init__(self, model_paths, providers=None):
        """
        Initialize multiple ONNX models
        
        Args:
            model_paths: List of paths to ONNX model files
            providers: List of execution providers (default: CUDA then CPU)
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        self.model_paths = model_paths
        self.sessions = []
        self.model_names = []
        
        # Load all models
        for model_path in model_paths:
            try:
                sess = ort.InferenceSession(model_path, providers=providers)
                self.sessions.append(sess)
                # Extract model name from path
                model_name = Path(model_path).stem
                self.model_names.append(model_name)
                print(f"✓ Loaded model: {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_path}: {e}")
        
        if not self.sessions:
            raise ValueError("No models could be loaded successfully")
    
    def run_inference(self, input_tensor):
        """Run inference on all models and return results with timing"""
        results = []
        timings = []
        
        for i, sess in enumerate(self.sessions):
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            
            try:
                t0 = time.time()
                output = sess.run([output_name], {input_name: input_tensor})[0]
                dt = (time.time() - t0) * 1000
                
                results.append(output)
                timings.append(dt)
                print(f"  {self.model_names[i]}: {dt:.2f}ms")
                
            except Exception as e:
                print(f"  {self.model_names[i]}: FAILED - {e}")
                # Add placeholder for failed inference
                results.append(None)
                timings.append(float('inf'))
        
        return results, timings


def create_comparison_image(orig_img, model_outputs, model_names, clean_img=None, target_height=512):
    """Create a horizontally stacked comparison image"""
    images_to_stack = []
    labels = []
    
    # Resize original image to target height while maintaining aspect ratio
    h, w = orig_img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    orig_resized = cv2.resize(orig_img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Add original image
    images_to_stack.append(add_text_label(orig_resized, "Original"))
    
    # Add model outputs
    for i, (output, name) in enumerate(zip(model_outputs, model_names)):
        if output is not None:
            processed_img = postprocess(output)
            img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
            img_resized = reinhard_color_transfer(img_resized, clean_img)
            images_to_stack.append(add_text_label(img_resized, name))
        else:
            # Create error placeholder
            error_img = np.zeros((target_height, new_w, 3), dtype=np.uint8)
            error_img = add_text_label(error_img, f"{name} - ERROR")
            images_to_stack.append(error_img)
    
    # Add clean/ground truth image if available
    if clean_img is not None:
        clean_resized = cv2.resize(clean_img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        images_to_stack.append(add_text_label(clean_resized, "Ground Truth"))
    
    # Stack horizontally
    comparison_6 = np.hstack(images_to_stack[:6])
    comparison_12 = np.hstack(images_to_stack[6:])
    comparison = np.vstack((comparison_6, comparison_12))
    return comparison

def reinhard_color_transfer(source, target):
    """
    Performs Reinhard color transfer from a source image to a target image.

    Args:
        source (np.ndarray): The source image in BGR format.
        target (np.ndarray): The target image in BGR format.

    Returns:
        np.ndarray: The color-corrected source image in BGR format.
    """
    # 1. Convert images from BGR to Lab color space.
    #    Lab space separates color (a, b) from lightness (L), which is ideal.
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # 2. Split the channels for both images.
    (s_l, s_a, s_b) = cv2.split(source_lab)
    (t_l, t_a, t_b) = cv2.split(target_lab)

    # 3. Compute the mean and standard deviation for each channel.
    (s_l_mean, s_l_std) = (s_l.mean(), s_l.std())
    (s_a_mean, s_a_std) = (s_a.mean(), s_a.std())
    (s_b_mean, s_b_std) = (s_b.mean(), s_b.std())

    (t_l_mean, t_l_std) = (t_l.mean(), t_l.std())
    (t_a_mean, t_a_std) = (t_a.mean(), t_a.std())
    (t_b_mean, t_b_std) = (t_b.mean(), t_b.std())
    
    # 4. Subtract the source mean from the source channels.
    s_l -= s_l_mean
    s_a -= s_a_mean
    s_b -= s_b_mean

    # 5. Scale by the ratio of standard deviations (add a small epsilon to avoid division by zero).
    s_l = (t_l_std / (s_l_std + 1e-6)) * s_l
    s_a = (t_a_std / (s_a_std + 1e-6)) * s_a
    s_b = (t_b_std / (s_b_std + 1e-6)) * s_b

    # 6. Add the target mean.
    s_l += t_l_mean
    s_a += t_a_mean
    s_b += t_b_mean

    # 7. Clip values to be within the valid range for L*a*b* (0-255 for uint8).
    s_l = np.clip(s_l, 0, 255)
    s_a = np.clip(s_a, 0, 255)
    s_b = np.clip(s_b, 0, 255)

    # 8. Merge the channels back and convert to an 8-bit unsigned integer.
    transfer_lab = cv2.merge([s_l, s_a, s_b]).astype("uint8")

    # 9. Convert back from Lab to BGR color space.
    transfer_bgr = cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2BGR)
    
    return transfer_bgr


def main():
    clean_img_dir = '/imgarc/nila/data/Super_Res/all_data/full_fov_and_wbc_patch_iter_3/organised_data/valid/clean'
    test_img_dir = '/imgarc/nila/data/Super_Res/all_data/full_fov_and_wbc_patch_iter_3/organised_data/valid/sr_1'
    out_dir = '/imgarc/nila/data/Deblur_Defocus/inference_all_models_v2_to_v9'
    model_paths = sorted(glob("/imgarc/nila/data/Deblur_Defocus/Models/*.onnx"))
    
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    try:
        # Initialize multi-model inference
        multi_model = MultiModelInference(model_paths)
        print(f"\n🚀 Loaded {len(multi_model.sessions)} models successfully\n")
        
        # Process each test image
        for i, image_path in enumerate(glob(f"{test_img_dir}/*.png")):
            print(f"Processing image {i+1}: {os.path.basename(image_path)}")
            
            # Load clean image if available
            clean_img_path = f"{clean_img_dir}/{os.path.basename(image_path)}"
            clean_img = None
            if os.path.exists(clean_img_path):
                clean_img = cv2.imread(clean_img_path, cv2.IMREAD_COLOR)
            
            # Preprocess input
            input_tensor, orig_bgr = load_and_preprocess(image_path)
            
            # Run inference on all models
            print("  Running inference...")
            model_outputs, timings = multi_model.run_inference(input_tensor)
            
            # Create comparison image
            comparison_img = create_comparison_image(
                orig_bgr, model_outputs, multi_model.model_names, clean_img, target_height=clean_img.shape[0]
            )
            
            
            # Print timing summary
            print("  Timing summary:")
            for name, timing in zip(multi_model.model_names, timings):
                if timing != float('inf'):
                    print(f"    {name}: {timing:.2f}ms")
                else:
                    print(f"    {name}: FAILED")
            

            # Save comparison image
            save_path = os.path.join(out_dir, f"comparison_{os.path.basename(image_path)}")
            cv2.imwrite(save_path, comparison_img)
            print(f"Saved comparison to: {save_path}")

            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()