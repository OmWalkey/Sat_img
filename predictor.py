from PIL import Image
import torch

def generate_caption(image, processor, model):
    prompt = "Describe the image."
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.tokenizer.decode(output[0], skip_special_tokens=True)

def answer_question(image, question, processor, model):
    inputs = processor(question, image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.tokenizer.decode(output[0], skip_special_tokens=True)
