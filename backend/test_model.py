import pandas as pd
from predict import predict
import random

def test_model():
    # Load a sample of training data
    try:
        fake_df = pd.read_csv("../train_model/data/Fake.csv")
        true_df = pd.read_csv("../train_model/data/True.csv")
        
        # Sample 5 fake and 5 true articles
        fake_sample = fake_df.sample(n=5)
        true_sample = true_df.sample(n=5)
        
        print("\nTesting Model Predictions:\n")
        print("-" * 50)
        
        # Test fake news articles
        print("\nTesting Fake News Articles:")
        for _, row in fake_sample.iterrows():
            text = f"{row['title']} {row['text']}"
            result = predict(text)
            print(f"\nTitle: {row['title'][:100]}...")
            print(f"Predicted: {result['result']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
        # Test true news articles
        print("\nTesting True News Articles:")
        for _, row in true_sample.iterrows():
            text = f"{row['title']} {row['text']}"
            result = predict(text)
            print(f"\nTitle: {row['title'][:100]}...")
            print(f"Predicted: {result['result']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_model()