# model_classes/blenderbot_model.py
from model_classes.base_model import BaseModel
import os
import gradio as gr
import sounddevice as sd
import numpy as np
from transformers import pipeline
from dotenv import load_dotenv
import os
# from model_classes.helper import load_image_from_url, render_results_in_image  # Import korrigiert
from .helper import load_image_from_url, render_results_in_image, summarize_predictions_natural_language
## course chapter
# https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/7/automatic-speech-recognition
# second part of the course chapter

class ObjectDetectionModel(BaseModel):
    def __init__(self):
        load_dotenv()
        port = os.environ.get('PORT1')

        self.od_pipe = pipeline("object-detection", model="facebook/detr-resnet-50") 

        demo = gr.Interface(
            fn=self.get_pipeline_prediction,
            inputs=gr.Image(label="Input image", type="pil"),
            outputs=gr.Image(label="Output image with predicted instances", type="pil")
        )

        demo.launch(share=True, server_port=int(os.environ['PORT1']))
        demo.close()

    def generate_response(self, msg: str):
        print(msg)

    def get_pipeline_prediction(self, pil_image):
        pipeline_output = self.od_pipe(pil_image)
        processed_image = render_results_in_image(pil_image, pipeline_output)
        return processed_image
