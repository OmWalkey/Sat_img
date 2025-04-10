import os
import cv2
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.plot import reshape_as_image
from skimage import exposure
from skimage.restoration import denoise_bilateral

RAW_IMAGE_DIR = r'C:\Sat_img\bhuvan_sat\train_image'
PROCESSED_DIR = 'data/processed_images'
NDVI_DIR = 'data/ndvi_images'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(NDVI_DIR, exist_ok=True)

# Helper: Histogram Matching for Radiometric Correction
def apply_histogram_equalization(image):
    image = image.astype(np.float32)
    image = image / image.max()  # Normalize before CLAHE
    return exposure.equalize_adapthist(image, clip_limit=0.03)

# Helper: Denoising
def apply_denoising(image):
    return denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15, channel_axis=-1)

# Helper: Resize and normalize
def resize_and_normalize(image, size=(512, 512)):
    image = cv2.resize(image, size)
    return image

# Helper: Band Selection (for NDVI or RGB)
def select_bands(image_path):
    with rasterio.open(image_path) as src:
        band_count = src.count
        if band_count >= 4:
            red = src.read(3).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            green = src.read(2).astype(np.float32)
            rgb_nir = np.stack([red, green, nir], axis=-1)
            rgb_nir = reshape_as_image(rgb_nir)
            return red, nir, rgb_nir
        elif band_count == 3:
            image = src.read().astype(np.float32)
            image = reshape_as_image(image)
            return None, None, image
        else:
            return None, None, None

# Helper: NDVI Calculation
def calculate_ndvi(red, nir):
    ndvi = (nir - red) / (nir + red + 1e-5)
    ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)  # scale to 0–255
    return ndvi_normalized

# Process a single image file
def preprocess_image(image_path, filename):
    red, nir, image = select_bands(image_path)
    if image is None:
        return None

    # Step 1: Radiometric Correction
    corrected_image = apply_histogram_equalization(image)

    # Step 2: Denoising
    denoised_image = apply_denoising(corrected_image)

    # Step 3: Resize
    resized_image = resize_and_normalize(denoised_image)

    # Normalize to 8-bit for saving
    clipped = np.clip(resized_image, 0, 1)
    final_image = (clipped * 255).astype(np.uint8)

    # Step 4: NDVI if applicable
    if red is not None and nir is not None:
        ndvi_image = calculate_ndvi(red, nir)
        ndvi_resized = resize_and_normalize(ndvi_image)
        cv2.imwrite(os.path.join(NDVI_DIR, filename), ndvi_resized)

    return final_image

# Preprocess Dataset
def preprocess_dataset():
    for filename in tqdm(os.listdir(RAW_IMAGE_DIR)):
        if filename.endswith(('.tif', '.tiff', '.jpg', '.png')):
            input_path = os.path.join(RAW_IMAGE_DIR, filename)
            output_path = os.path.join(PROCESSED_DIR, filename)
            processed = preprocess_image(input_path, filename)
            if processed is not None:
                cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    preprocess_dataset()
    print("Preprocessing and NDVI generation completed for Bhuvan satellite data.")
