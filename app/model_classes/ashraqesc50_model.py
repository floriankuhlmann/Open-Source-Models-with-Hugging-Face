# model_classes/blenderbot_model.py
from datasets import load_dataset, load_from_disk
from datasets import Audio
from model_classes.base_model import BaseModel
# from IPython.display import Audio as IPythonAudio
from transformers import pipeline
import sounddevice as sd
import numpy as np

class AshraqEsc50Model(BaseModel):
    def __init__(self):

        # This dataset is a collection of different sounds of 5 seconds
        dataset = load_dataset("ashraq/esc50",
                              split="train[0:10]")
        # dataset = load_from_disk("./models/ashraq/esc50/train")
        audio_sample = dataset[0]
        self.generate_response("audio samlpe:")
        self.generate_response(audio_sample)


        # IPythonAudio(audio_sample["audio"]["array"],
        #      rate=audio_sample["audio"]["sampling_rate"])
        
        print(f"Playing audio:")
        self.play_audio(audio_sample["audio"]["array"], audio_sample["audio"]["sampling_rate"])

        zero_shot_classifier = pipeline(
            task="zero-shot-audio-classification",
            model="laion/clap-htsat-unfused")

        self.generate_response(zero_shot_classifier.feature_extractor.sampling_rate)
        self.generate_response(audio_sample["audio"]["sampling_rate"])
        dataset = dataset.cast_column(
                    "audio",
                    Audio(sampling_rate=48_000))
        audio_sample = dataset[0]
        self.generate_response(audio_sample)
        candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]
        self.generate_response(zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels))
        candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a bird singing",
                    "Sound of an airplane"]
        self.generate_response(zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels))
        # Fügen Sie eine weitere Nachricht zur Konversation hinzu

    def generate_response(self, msg: str):
        print(msg)

    def play_audio(self, audio_array, sampling_rate):
        sd.play(audio_array, sampling_rate)
        sd.wait()  # Warten, bis die Wiedergabe abgeschlossen ist