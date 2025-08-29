import time
import cv2
import numpy as np
import onnxruntime as ort
from glob import glob
import os


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
    out_dir = '/imgarc/nila/data/Deblur_Defocus/inference_6_and_2_blocks_patches'
    model_path_2_blocks = '/home/as76usr/sigtuple/Eshwar/Garuda-model-dev/AS76/ISP/Defocus-Deblur/netG_A_dynamic_res_2_blocks.onnx'
    model_path_6_blocks = "/home/as76usr/sigtuple/Eshwar/Garuda-model-dev/AS76/ISP/Defocus-Deblur/netG_A_dynamic_res_6blocks.onnx"
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Create two separate sessions
    sess_2_blocks = ort.InferenceSession(model_path_2_blocks, providers=providers)
    sess_6_blocks = ort.InferenceSession(model_path_6_blocks, providers=providers)

    # Get input/output names for both models
    input_name_2 = sess_2_blocks.get_inputs()[0].name
    output_name_2 = sess_2_blocks.get_outputs()[0].name

    input_name_6 = sess_6_blocks.get_inputs()[0].name
    output_name_6 = sess_6_blocks.get_outputs()[0].name

    for image_path in glob(f"{test_img_dir}/*.png"):
        try:
            clean_img = cv2.imread(f"{clean_img_dir}/{os.path.basename(image_path)}", cv2.IMREAD_COLOR)
            in_img, orig_bgr = load_and_preprocess(image_path)
            
            # Inference with 9-blocks model
            t1 = time.time()
            y_2_blocks = sess_2_blocks.run([output_name_2], {input_name_2: in_img})[0]
            t2 = time.time()
            time_2_blocks = round((t2 - t1) * 1000, 2)
            
            # Inference with 6-blocks model
            t3 = time.time()
            y_6_blocks = sess_6_blocks.run([output_name_6], {input_name_6: in_img})[0]
            t4 = time.time()
            time_6_blocks = round((t4 - t3) * 1000, 2)
            
            print(f"{os.path.basename(image_path)}: 2-blocks={time_2_blocks}ms, 6-blocks={time_6_blocks}ms")

            # Process 9-blocks output (your original processing)
            out_rgb_2= postprocess(y_2_blocks)
            out_bgr_2= cv2.cvtColor(out_rgb_2, cv2.COLOR_RGB2BGR)
            out_bgr_2= cv2.resize(out_bgr_2, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            # out_bgr_transferred_2= reinhard_color_transfer(source=out_bgr_2, target=clean_img)
            out_bgr_transferred_2= out_bgr_2
            
            # Process 6-blocks output
            out_rgb_6 = postprocess(y_6_blocks)
            out_bgr_6 = cv2.cvtColor(out_rgb_6, cv2.COLOR_RGB2BGR)
            out_bgr_6 = cv2.resize(out_bgr_6, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            # out_bgr_transferred_6 = reinhard_color_transfer(source=out_bgr_6, target=clean_img)
            out_bgr_transferred_6 = out_bgr_6
            
            # Concatenate in the order you specified: orig_bgr, 6-blocks output, 9-blocks output, clean_img
            result = np.hstack([orig_bgr, out_bgr_transferred_6, out_bgr_transferred_2, clean_img])
            cv2.imwrite(f"{out_dir}/{os.path.basename(image_path)}", result)
            
        except Exception as e:
            print(f"Error processing image {image_path} " + str(e))
        


if __name__ == "__main__":
    main()