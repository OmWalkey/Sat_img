import os
from PIL import Image
import numpy as np

def tile_image(image_path, output_dir, tile_size=256):
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    tile_id = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img_array[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                Image.fromarray(tile).save(os.path.join(output_dir, f"{base_name}_tile{tile_id}.png"))
                tile_id += 1

if __name__ == "__main__":
    for img in os.listdir("processed"):
        if img.endswith(".png"):
            tile_image(os.path.join("processed", img), "tiles")
