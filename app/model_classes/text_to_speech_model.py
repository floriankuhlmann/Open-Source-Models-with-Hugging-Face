# model_classes/blenderbot_model.py
from model_classes.base_model import BaseModel
import os
import gradio as gr
import sounddevice as sd
import numpy as np
from transformers import pipeline
from dotenv import load_dotenv
import pkg_resources

## course chapter
# https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/7/automatic-speech-recognition
# second part of the course chapter

class TextToSpeechModel(BaseModel):
    def __init__(self):
        load_dotenv()
        # narrator = pipeline("text-to-speech",
        #             model="./models/kakao-enterprise/vits-ljs")
        
        # Use a pipeline as a high-level helper
        narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")

        text = """
            Researchers at the Allen Institute for AI, \
            HuggingFace, Microsoft, the University of Washington, \
            Carnegie Mellon University, and the Hebrew University of \
            Jerusalem developed a tool that measures atmospheric \
            carbon emitted by cloud servers while training machine \
            learning models. After a model’s size, the biggest variables \
            were the server’s location and time of day it was active.
            """

        narrated_text = narrator(text)
        self.play_audio(narrated_text["audio"][0],narrated_text["sampling_rate"])

    def generate_response(self, msg: str):
        print(msg)

    def play_audio(self, audio_array, sampling_rate):
        sd.play(audio_array, sampling_rate)
        sd.wait()  # Wait until the audio has finished playing
    