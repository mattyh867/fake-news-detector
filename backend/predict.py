from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load tokenizer from local directory
tokenizer_path = os.path.join(current_dir, "model", "tokenizer")
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

# Load model from Hugging Face Hub
model_name = "mattyh867/fake-news-detector"
try:
    model = RobertaForSequenceClassification.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model from Hugging Face: {e}")
    print("Falling back to local model") 
    model_path = os.path.join(current_dir, "model")
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
    