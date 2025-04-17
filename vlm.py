import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os
from tqdm import tqdm

# Use Qwen 2.5 Base model
MODEL_NAME = "Qwen/Qwen-VL-Chat-Int4"

# Load model and processor
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Directory containing the preprocessed images
IMAGE_DIR = "processed"
OUTPUT_FILE = "vlm_output.txt"

# VLM inference function
def vlm_inference(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=256)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return result

# Run inference on all images in directory
def run_batch_vlm():
    results = {}
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))]
    for file in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, file)
        output = vlm_inference(img_path)
        results[file] = output

    with open(OUTPUT_FILE, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    run_batch_vlm()
