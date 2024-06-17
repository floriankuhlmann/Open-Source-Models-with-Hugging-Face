# model_classes/blenderbot_model.py
from model_classes.base_model import BaseModel
import os
import gradio as gr
import sounddevice as sd
import numpy as np
from transformers import pipeline
from dotenv import load_dotenv
import os

## course chapter
# https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/7/automatic-speech-recognition
# second part of the course chapter

class AutomaticSpeechRecognition2Model(BaseModel):
    def __init__(self):
        load_dotenv()
        port = os.environ.get('PORT1')
        ## the following code from the course is wrong!
        # self.asr = pipeline(task="automatic-speech-recognition", model="./models/distil-whisper/distil-small.en")
        # https://huggingface.co/distil-whisper/distil-small.en?library=transformers
        self.asr = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en") 

        demo = gr.Blocks()
        self.generate_response("Data preparation")
        mic_transcribe = gr.Interface(
            fn=self.transcribe_speech,
            inputs=gr.Audio(sources="microphone", type="filepath"),
            outputs=gr.Textbox(label="Transcription",lines=3),
            allow_flagging="never"
            )

        file_transcribe = gr.Interface(
            fn=self.transcribe_speech,
            inputs=gr.Audio(sources="upload",type="filepath"),
            outputs=gr.Textbox(label="Transcription",lines=3),
            allow_flagging="never",
        )

        with demo:
            gr.TabbedInterface(
            [mic_transcribe,
            file_transcribe],
            ["Transcribe Microphone",
            "Transcribe Audio File"],
            )

        demo.launch(share=True, server_port=int(port))
        demo.close()

    def generate_response(self, msg: str):
        print(msg)

    def play_audio(self, audio_array, sampling_rate):
        sd.play(audio_array, sampling_rate)
        sd.wait()  # Warten, bis die Wiedergabe abgeschlossen ist
    
    def transcribe_speech(self, filepath):
        if filepath is None:
            gr.Warning("No audio found, please retry.")
            return ""
        output = self.asr(filepath)
        return output["text"]