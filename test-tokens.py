import transformers
import torch
import time
from datasets import load_dataset
import argparse

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for inference.")

model_path = "models/OpenHermes-2-Mistral-7B"

# Initialize the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto"
)

# Example text (you can replace this with your actual inputs)
example_texts = ["This is a test sentence.", "Another short example."]

# Function to measure tokens per second
def measure_tokens_per_second(texts, pipeline):
    total_tokens = 0
    start_time = time.time()

    for text in texts:
        # Count tokens in the input
        tokens = pipeline.tokenizer.encode(text)
        total_tokens += len(tokens)

        # Generate output (you can add your generation configuration here)
        pipeline(text)

    end_time = time.time()
    duration = end_time - start_time
    tokens_per_second = total_tokens / duration
    return tokens_per_second, duration

tokens_per_second, duration = measure_tokens_per_second(example_texts, pipeline)

print(f"Processed {len(example_texts)} texts in {duration:.2f} seconds.")
print(f"Tokens per second: {tokens_per_second:.2f}")

