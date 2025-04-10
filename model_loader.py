import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# Load LLaVA (Large Language and Vision Assistant)
def load_vlm_model():
    processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model.eval()
    return processor, model
