# from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def download_model(model_name, save_directory):
    # Modell und Tokenizer laden
    model = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Modell und Tokenizer speichern
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

if __name__ == "__main__":
    print(f"start skript")
    parser = argparse.ArgumentParser(description="Wähle ein Modell zum Download.")
    parser.add_argument('--model', type=str, required=True, help="Name des zu ladenden Modells (z.B. 'blenderbot')")
    args = parser.parse_args()

    if args.model == "blenderbot":
        model_name = "facebook/blenderbot-400M-distill"
        save_directory = "../app/models/facebook/blenderbot-400M-distill"
    elif args.model == "nllb200":
        model_name = "facebook/nllb-200-distilled-600M"
        save_directory = "../app/models/facebook/nllb-200-distilled-600M"
    elif args.model == "bart":
        model_name = "facebook/bart-large-cnn"
        save_directory = "../app/models/facebook/bart-large-cnn"
    else:
        raise ValueError(f"Unbekanntes Modell: {args.model}")

    download_model(model_name, save_directory)