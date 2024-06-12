import argparse
from model_classes.gradiospeech_model import GradioSpeechModel
from model_classes.librispeech_model import LibrispeechModel
from model_classes.ashraqesc50_model import AshraqEsc50Model
from model_classes.minilml6_model import MiniLmL6Model
from model_classes.nllb200_model import Nllb200Model
from model_classes.blenderbot_model import BlenderBotModel
from transformers.utils import logging

def main(model_name: str):
    print("Hello, let's play with a hugging face!")
    logging.set_verbosity_error()

    # Wählen Sie das Modell basierend auf dem Parameter
    if model_name == "blenderbot":
        model = BlenderBotModel()
    elif model_name == "nllb200":
        model = Nllb200Model()
    elif model_name == "minilml6":
        model = MiniLmL6Model() 
    elif model_name == "ashraqesc50":
        model = AshraqEsc50Model()
    elif model_name == "librispeech":
        ## course chapter
        # https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/7/automatic-speech-recognition
        # first part of the course chapter
        model = LibrispeechModel()
    elif model_name == "gradiospeech":
        ## course chapter
        # https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/7/automatic-speech-recognition
        # second part of the course chapter
        model = GradioSpeechModel()
    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wähle ein Modell zum Testen und Nutzen.")
    parser.add_argument('--model', type=str, required=True, help="Name des zu nutzenden Modells (z.B. 'blenderbot')")
    args = parser.parse_args()
    main(args.model)