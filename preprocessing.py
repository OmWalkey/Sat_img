import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

# --- Paths ---
INPUT_DIR = r"C:\Sat_img\MASATI-v2\coast_ship"
OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Enhancement Functions ---

def denoise_image(img):
    """
    Apply mild Non-Local Means Denoising.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)

def gamma_correction(img, gamma=1.1):
    """
    Apply gamma correction to slightly brighten image and improve contrast.
    """
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def sharpen_image(img):
    """
    Apply very gentle sharpening to enhance edges without overshooting.
    """
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    return cv2.filter2D(img, -1, kernel)

def enhance_image(img):
    """
    Apply full enhancement pipeline: Denoise + Gamma Correction + Sharpen.
    """
    img = denoise_image(img)
    img = gamma_correction(img, gamma=1.1)
    img = sharpen_image(img)
    return img

def radiometric_correction(img):
    """
    Placeholder for radiometric correction.
    (This requires sensor metadata; skip for now)
    """
    return img

def geometric_correction(input_path, output_path):
    """
    Reproject image to standard coordinate reference system (EPSG:4326).
    """
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest)

# --- Preprocessing Workflow ---
def preprocess():
    for filename in tqdm(os.listdir(INPUT_DIR)):
        if filename.lower().endswith(('.png', '.jpg', '.tif', '.tiff')):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)

            # Read image
            img = cv2.imread(input_path)
            if img is None:
                print(f"Skipping unreadable file: {filename}")
                continue

            # Radiometric Correction (if needed)
            img = radiometric_correction(img)

            # Enhancement
            img = enhance_image(img)

            # Save intermediate result
            temp_path = os.path.join(OUTPUT_DIR, f"temp_{filename}")
            cv2.imwrite(temp_path, img)

            # Geometric correction (only for GeoTIFF)
            if filename.lower().endswith(('.tif', '.tiff')):
                try:
                    with rasterio.open(input_path) as src:
                        if src.crs:
                            geometric_correction(temp_path, output_path)
                            os.remove(temp_path)
                        else:
                            print(f"Skipping geometric correction for {filename}: No CRS found.")
                            os.rename(temp_path, output_path)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    os.rename(temp_path, output_path)
            else:
                # Non-georeferenced image, just move the result
                os.rename(temp_path, output_path)


if __name__ == "__main__":
    preprocess()
