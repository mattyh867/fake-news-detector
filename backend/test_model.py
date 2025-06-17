import pandas as pd
from predict import predict

def test_model():
    try:
        # Load the test dataset
        test_df = pd.read_csv("../train_model/data/Test.csv")
        
        print("\nTesting Model with Test Dataset:\n")
        print("-" * 50)
        
        # Test all articles in the test dataset
        for _, row in test_df.iterrows():
            # Combine title and text if both exist
            text_parts = []
            if 'title' in row:
                text_parts.append(str(row['title']))
            if 'text' in row:
                text_parts.append(str(row['text']))
            
            text = " ".join(text_parts)
            result = predict(text)
            
            print(f"\nTitle: {row.get('title', 'No title')[:100]}...")
            print(f"Predicted: {result['result']}")
            print(f"Confidence: {result['confidence']:.2%}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("\nAvailable columns in Test.csv:")
        print(test_df.columns)

if __name__ == "__main__":
    test_model()