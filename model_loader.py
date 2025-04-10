from llava.model.builder import load_pretrained_model

def load_vlm_model():
    model_path = "liuhaotian/llava-v1.5-7b"  # You can change this to a local path if needed

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_path, model_name="llava"
    )

    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "context_len": context_len
    }
