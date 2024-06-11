import argparse
from model_classes.minilml6_model import MiniLmL6Model
from model_classes.nllb200_model import Nllb200Model
from model_classes.blenderbot_model import BlenderBotModel
from transformers.utils import logging

def main(model_name: str):
    print("Hello, let's play!")
    logging.set_verbosity_error()

    # Wählen Sie das Modell basierend auf dem Parameter
    if model_name == "blenderbot":
        model = BlenderBotModel()
    elif model_name == "nllb200":
        model = Nllb200Model()
    elif model_name == "minilml6":
        model = MiniLmL6Model() 
    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wähle ein Modell zum Testen und Nutzen.")
    parser.add_argument('--model', type=str, required=True, help="Name des zu nutzenden Modells (z.B. 'blenderbot')")
    args = parser.parse_args()
    main(args.model)