# model_classes/blenderbot_model.py
from transformers import pipeline, Conversation, AutoTokenizer, BlenderbotForConditionalGeneration
from model_classes.base_model import BaseModel

class BlenderBotModel(BaseModel):
    def __init__(self):

        model_path = "./models/facebook/blenderbot-400M-distill"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_path)
        print("Model loaded:", model_path)

        self.chatbot = pipeline(task="conversational", model=self.model, tokenizer=self.tokenizer)
    
        user_message = "What are some fun activities I can do in the winter?"
        self.generate_response(user_message)

        # Fügen Sie eine weitere Nachricht zur Konversation hinzu
        user_message = "What else do you recommend?"
        self.generate_response(user_message)

    def generate_response(self, user_message: str):
        conversation = Conversation(user_message)
        response = self.chatbot(conversation)
        print(response)