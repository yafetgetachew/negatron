import numpy as np
import cv2


def detect_base_color(img):
    """
    Detect the base (darkest) color in the image by analyzing the darkest 10% of pixels.

    Parameters
    ----------
    img : ndarray
        Input image in BGR format.

    Returns
    -------
    ndarray
        Mean color (B, G, R) of the darkest 10% pixels.
    """
    pixels = img.reshape(-1, 3)
    sorted_pixels = pixels[np.argsort(np.sum(pixels, axis=1))]
    darkest = sorted_pixels[: len(pixels) // 10]
    return np.mean(darkest, axis=0)


def convert_negative(img, base_color):
    """
    Convert the given image to a negative using the specified base color.

    The image is normalized to the range [0, 1], inverted, and each channel
    is adjusted based on the base color. Finally, the result is scaled back
    to [0, 255].

    Parameters
    ----------
    img : ndarray
        Input image in BGR format.
    base_color : tuple or ndarray
        Base color (B, G, R) for conversion.

    Returns
    -------
    ndarray
        Negative image in BGR format.
    """
    img_float = img.astype(float) / 255.0
    base_float = np.array(base_color) / 255.0

    result = 1.0 - img_float
    for channel in range(3):
        result[:, :, channel] *= 1.0 / (1.0 - base_float[channel])
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def apply_adjustments(image, hue=0, saturation=100, contrast=100, brightness=100,
                      shadows=100, highlights=100):
    """
    Apply image adjustments including hue, saturation, contrast, brightness,
    shadows, and highlights.

    The adjustments are applied in the following order:
      1. Convert to HSV and adjust hue and saturation.
      2. Convert to LAB and adjust shadows and highlights.
      3. Adjust contrast.
      4. Adjust brightness.

    Parameters
    ----------
    image : ndarray
        Input image in BGR format.
    hue : int, optional
        Hue adjustment in degrees (default is 0).
    saturation : int, optional
        Saturation percentage (default is 100).
    contrast : int, optional
        Contrast percentage (default is 100).
    brightness : int, optional
        Brightness percentage (default is 100).
    shadows : int, optional
        Shadows adjustment percentage (default is 100).
    highlights : int, optional
        Highlights adjustment percentage (default is 100).

    Returns
    -------
    ndarray
        Adjusted image in BGR format.
    """
    adjusted_image = image.copy()

    if hue != 0 or saturation != 100:
        hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        if hue != 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
        if saturation != 100:
            hsv[:, :, 1] *= (saturation / 100.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        adjusted_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if shadows != 100 or highlights != 100:
        lab = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        shadow_mask = np.clip((127 - l_channel) / 127, 0, 1)
        highlight_mask = np.clip((l_channel - 127) / 127, 0, 1)
        if shadows != 100:
            shadow_factor = shadows / 100.0
            l_channel += (shadow_factor - 1) * shadow_mask * l_channel
        if highlights != 100:
            highlight_factor = highlights / 100.0
            l_adjustment = (highlight_factor - 1) * highlight_mask * (255 - l_channel)
            l_channel += l_adjustment
        l_channel = np.clip(l_channel, 0, 255)
        lab[:, :, 0] = l_channel
        adjusted_image = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    if contrast != 100:
        contrast_factor = contrast / 100.0
        adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=contrast_factor, beta=0)

    if brightness != 100:
        brightness_offset = (brightness - 100) * 1.27  # Rough scaling factor, is it right?
        adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=1, beta=brightness_offset)

    return adjusted_image
