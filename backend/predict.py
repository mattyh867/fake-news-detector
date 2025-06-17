from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

def predict(text):
    # Load model from Hugging Face with authentication
    model = RobertaForSequenceClassification.from_pretrained(
        "mattyh867/fake-news-detector",
        use_auth_token=True
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        "mattyh867/fake-news-detector",
        use_auth_token=True
    )
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {"result": "FAKE" if predicted_class == 1 else "REAL", "confidence": confidence}
