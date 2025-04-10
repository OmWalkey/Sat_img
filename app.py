import gradio as gr
from PIL import Image
from vlm.model_loader import load_vlm_model
from vlm.predictor import generate_caption, answer_question

processor, model = load_vlm_model()

def process_image(image, question):
    caption = generate_caption(image, processor, model)
    answer = answer_question(image, question, processor, model)
    return caption, answer

demo = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Your Question")],
    outputs=[gr.Textbox(label="Caption"), gr.Textbox(label="Answer")],
    title="Satellite VLM GUI",
    description="Upload a satellite image and ask questions about the scene."
)

if __name__ == "__main__":
    demo.launch()
