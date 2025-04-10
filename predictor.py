from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import torch

def generate_caption(vlm, image):
    # Prepare image
    image_tensor = process_images([image], vlm["image_processor"])
    image_tensor = image_tensor.to(vlm["model"].device, dtype=torch.float16)

    # Set up prompt
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], "Describe the scene in the image.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        vlm["tokenizer"],
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(vlm["model"].device)

    output_ids = vlm["model"].generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=100,
        use_cache=True
    )

    output = vlm["tokenizer"].decode(output_ids[0], skip_special_tokens=True)
    return output.strip()

def answer_question(vlm, image, question):
    # Prepare image
    image_tensor = process_images([image], vlm["image_processor"])
    image_tensor = image_tensor.to(vlm["model"].device, dtype=torch.float16)

    # Set up prompt
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        vlm["tokenizer"],
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(vlm["model"].device)

    output_ids = vlm["model"].generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=100,
        use_cache=True
    )

    output = vlm["tokenizer"].decode(output_ids[0], skip_special_tokens=True)
    return output.strip()
