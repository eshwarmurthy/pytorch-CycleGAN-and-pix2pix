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


def main():

    clean_img_dir = '/media/adithya/home_data/SR/Dataset_v12/valid/clean_denoised'
    test_img_dir = '/media/adithya/home_data/SR/Dataset_v12/valid/sr_4_denoised'
    model_path = '/home/adithya/Code/pytorch-CycleGAN-and-pix2pix/netG_A_dynamic.onnx'
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    for image_path in glob(f"{test_img_dir}/*.png"):
        clean_img = cv2.imread(f"{clean_img_dir}/{os.path.basename(image_path)}", cv2.IMREAD_COLOR)
        in_img, orig_bgr = load_and_preprocess(image_path)
        t0 = time.time()
        y = sess.run([output_name], {input_name: in_img})[0]
        dt = (time.time() - t0) * 1000
        print(f"[i] Inference done in {dt:.2f} ms  |  Output tensor: {y.shape}")
        out_rgb = postprocess(y)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(out_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", np.hstack([orig_bgr, out_bgr, clean_img]))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
