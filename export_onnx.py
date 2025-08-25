"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def export_g_to_onnx(
                        G: torch.nn.Module,
                        onnx_path: str = "netG_A_dynamic.onnx",
                        sample_hw=(256, 256),
                        channels: int = 3,
                        opset: int = 13,
                    ):
    G.eval()
    dummy = torch.randn(1, channels, sample_hw[0], sample_hw[1], device="cuda" if next(G.parameters()).is_cuda else "cpu")

    torch.onnx.export(
        G,
        dummy,
        onnx_path,
        export_params=True, 
        opset_version=opset,        
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    )

    print(f"Exported → {onnx_path}")

def validate_onnx_dynamic(onnx_path: str):
    import onnx, numpy as np, onnxruntime as ort
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def run_random(bs, h, w):
        x = np.random.randn(bs, 3, h, w).astype(np.float32)
        x = (x - 0.5) / 0.5
        y, = sess.run(None, {"input": x})
        print(f"OK: input {x.shape} → output {y.shape}")

    for bs,h,w in [(1,256,256),(2,192,320),(4,512,384),(3,286,286)]:
        run_random(bs,h,w)


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # Export the generator model to ONNX
    onnx_path = "netG_A_dynamic.onnx"
    export_g_to_onnx(model.netG_A, onnx_path)

    # Validate the exported ONNX model
    validate_onnx_dynamic(onnx_path)
