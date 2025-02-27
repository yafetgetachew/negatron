import numpy as np

def detect_base_color(img):
    """
    Detects the base (darkest) color in the image by analyzing the darkest 10% of the pixels.
    TODO: Add other algos, but I think this one should work for most cases. Unless there's a Scanner
    that misbehaves
    """
    pixels = img.reshape(-1, 3)
    sorted_pixels = pixels[np.argsort(np.sum(pixels, axis=1))]
    darkest = sorted_pixels[:len(pixels) // 10]
    return np.mean(darkest, axis=0)

def convert_negative(img, base_color):
    """
    Converts the given image to a negative using the specified base color.
    """
    img_float = img.astype(float) / 255.0
    base_float = np.array(base_color) / 255.0

    result = 1.0 - img_float
    for channel in range(3):
        result[:, :, channel] = result[:, :, channel] * (1.0 / (1.0 - base_float[channel]))

    return np.clip(result * 255, 0, 255).astype(np.uint8)
