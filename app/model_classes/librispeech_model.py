# model_classes/blenderbot_model.py
from datasets import load_dataset, load_from_disk
from datasets import Audio
from model_classes.base_model import BaseModel
# from IPython.display import Audio as IPythonAudio
from transformers import pipeline
import sounddevice as sd
import numpy as np

class LibrispeechModel(BaseModel):
    def __init__(self):

        self.generate_response("Data preparation")
        dataset = load_dataset("librispeech_asr",
                       split="train.clean.100",
                       streaming=True,
                       trust_remote_code=True)
        example = next(iter(dataset))
        dataset_head = dataset.take(5)
        list(dataset_head)
        list(dataset_head)[2]
        self.generate_response(example)

        print("Playing audio:")
        self.play_audio(example["audio"]["array"], example["audio"]["sampling_rate"])

        self.generate_response("Build the pipeline")
        ## the following code from the course is wrong!
        # asr = pipeline(task="automatic-speech-recognition", model="./models/distil-whisper/distil-small.en")
        # https://huggingface.co/distil-whisper/distil-small.en?library=transformers        
        asr = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en") 
        asr.feature_extractor.sampling_rate
        self.generate_response(example['audio']['sampling_rate'])
        asr(example["audio"]["array"])
        self.generate_response(example["text"])  

    def generate_response(self, msg: str):
        print(msg)

    def play_audio(self, audio_array, sampling_rate):
        sd.play(audio_array, sampling_rate)
        sd.wait()  # Warten, bis die Wiedergabe abgeschlossen ist