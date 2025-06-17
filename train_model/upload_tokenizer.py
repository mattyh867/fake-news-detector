from huggingface_hub import HfApi
from transformers import RobertaTokenizer

def upload_tokenizer():
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Push tokenizer to hub
    tokenizer.push_to_hub("mattyh867/fake-news-detector", use_auth_token=True)
    
    print("Tokenizer uploaded successfully!")

if __name__ == "__main__":
    upload_tokenizer()
