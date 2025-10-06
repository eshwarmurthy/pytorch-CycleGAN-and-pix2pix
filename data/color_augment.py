import numpy as np
import cv2

def apply_hue_shift(img_np, hue_shift):
    """Applies a hue shift to a numpy image array (expects uint8)."""
    if img_np.dtype != np.uint8:
        raise TypeError("Input image for hue shift must be np.uint8")

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    # Hue in OpenCV is in [0, 179] for 8-bit images
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + hue_shift) % 180
    shifted_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return shifted_rgb.astype(np.uint8)

def apply_warm_tint(img_np, warmth=0.2):
    """Applies a warm tint to a numpy image array (expects uint8)."""
    # Work with floats for precision to avoid clipping issues with uint8 math
    img_warmed = img_np.astype(np.float32)

    # Add warmth to the Red channel
    img_warmed[..., 0] += warmth * 255
    # Reduce coolness from the Blue channel
    img_warmed[..., 2] -= (warmth / 2) * 255

    # Clip the values to the valid [0, 255] range and convert back to uint8
    return np.clip(img_warmed, 0, 255).astype(np.uint8)