import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from model_classes.base_model import BaseModel
import gc

class Nllb200Model(BaseModel):
    def __init__(self):
        
        model_path = "./models/facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print("Model loaded:", model_path)
        
        # Translator-Pipeline 
        self.translator = pipeline(task="translation", model=self.model, tokenizer=self.tokenizer, torch_dtype=torch.bfloat16)
    
        text = """\
            My puppy is adorable, \
            Your kitten is cute.
            Her panda is friendly.
            His llama is thoughtful. \
            We all have nice pets!"""
        text_translated = self.translator(text,src_lang="eng_Latn",tgt_lang="fra_Latn")       
        self.generate_response(text_translated)
        del self.translator
        gc.collect()
        self.summarizer = pipeline(task="summarization",
                      model="./models/facebook/bart-large-cnn",
                      torch_dtype=torch.bfloat16)
        text = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""
        self.summary = self.summarizer(text,
                     min_length=10,
                     max_length=100)
        self.generate_response(self.summary)

    def generate_response(self, user_message: str):
        print(user_message)