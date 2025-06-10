from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

import os

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model")
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {"result": "FAKE" if predicted_class == 1 else "REAL", "confidence": confidence}
    