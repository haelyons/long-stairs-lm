from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path="models/OpenHermes-2-Mistral-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    offload_folder="offload_weights",
    )
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, legacy=False)    # fast tokenizer 

# sampling parameters: llama-precise
gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 300,

}

messages = [
    {"role": "user", "content": "Good morning Mr. Bourdain! Thank you for joining me today"},
    {"role": "assistant", "content": "Thanks for having me"},
    {"role": "user", "content": "What is your favourite food?"}
]

prompt_tokenized=tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
prompt_tokenized=torch.tensor([prompt_tokenized]).to("cuda")

output_ids = model.generate(prompt_tokenized, **gen_config)

response=tokenizer.decode(output_ids[0])
print(response)
