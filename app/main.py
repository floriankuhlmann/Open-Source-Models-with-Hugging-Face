import argparse
from model_classes.object_detection_model import ObjectDetectionModel
from model_classes.automatic_speech_recognition_2_model import AutomaticSpeechRecognition2Model
from model_classes.automatic_speech_recognition_1_model import AutomaticSpeechRecognition1Model
from model_classes.zero_shot_audio_classification_model import ZeroShotAudioClassificationModel
from model_classes.sentence_embeddings_model import SentenceEmbeddingsModel
from model_classes.translation_and_summarization_model import TranslationandSummarizationModel
from model_classes.natural_language_processing_model import NaturalLanguageProcessingModel
from model_classes.text_to_speech_model import TextToSpeechModel
from transformers.utils import logging

def main(model_name: str):
    print("Hello, let's play with a hugging face!")
    logging.set_verbosity_error()

    # Wählen Sie das Modell basierend auf dem Parameter
    if model_name == "natural-language-processing":
        model = NaturalLanguageProcessingModel()
    elif model_name == "translation-and-summarization":
        model = TranslationandSummarizationModel()
    elif model_name == "sentence-embeddings":
        model = SentenceEmbeddingsModel() 
    elif model_name == "zero-shot-audio-classification":
        model = ZeroShotAudioClassificationModel()
    elif model_name == "automatic-speech-recognition-1":
        model = AutomaticSpeechRecognition1Model()
    elif model_name == "automatic-speech-recognition-2":
        model = AutomaticSpeechRecognition2Model()
    elif model_name == "text-to-speech":
        model = TextToSpeechModel()
    elif model_name == "object-detection":
        model = ObjectDetectionModel()
    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wähle ein Modell zum Testen und Nutzen.")
    parser.add_argument('--model', type=str, required=True, help="Name des zu nutzenden Modells (z.B. 'blenderbot')")
    args = parser.parse_args()
    main(args.model)