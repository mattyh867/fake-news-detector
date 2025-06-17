import pandas as pd
import random

# Read the original datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Sample 5 articles from each
fake_sample = fake_df.sample(n=5, random_state=42)
true_sample = true_df.sample(n=5, random_state=42)

# Combine samples
test_df = pd.concat([fake_sample, true_sample], ignore_index=True)

# Add label column (1 for fake, 0 for true)
test_df['label'] = [1]*5 + [0]*5

# Shuffle the combined dataset
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to Test.csv
test_df.to_csv("Test.csv", index=False)
