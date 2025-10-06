import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

# --- START: NEW IMPORTS ---
import numpy as np
from data.color_augment import apply_hue_shift, apply_warm_tint
# --- END: NEW IMPORTS ---


class AlignedDataset(BaseDataset):
    """A dataset class for paired image-to-image translation.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During training, you need to prepare data in '/path/to/data/train' folder.
    This dataset will make sure that image A and image B are loaded together.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert self.opt.load_size >= self.opt.crop_size  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # --- START: NEW AUGMENTATION CODE ---
        # The images A and B are now separate PIL.Image objects.
        # We will apply our consistent color augmentation here.
        
        # 1. Convert PIL images to numpy arrays
        A_np = np.array(A)
        B_np = np.array(B)

        # 2. Generate ONE consistent set of random augmentation parameters
        hue_shift = np.random.randint(-10, 11)  # e.g., from -10 to 10
        
        # 3. Apply the SAME hue shift to both A and B
        A_np_aug = apply_hue_shift(A_np, hue_shift)
        B_np_aug = apply_hue_shift(B_np, hue_shift)

        # 4. Apply the SAME warm tint to both A and B (with 10% probability)
        if np.random.random() < 0.1:
            warmth = np.random.uniform(0.1, 0.3)
            A_np_aug = apply_warm_tint(A_np_aug, warmth)
            B_np_aug = apply_warm_tint(B_np_aug, warmth)
            
        # 5. Convert augmented numpy arrays back to PIL images
        A = Image.fromarray(A_np_aug)
        B = Image.fromarray(B_np_aug)
        # --- END: NEW AUGMENTATION CODE ---

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)