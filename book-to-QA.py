# QA in title refers to Question / Answer format (for ChatML conversion in this case)
# Big thanks to: https://github.com/geronimi73/qlora-minimal/tree/main

import transformers
import evaluate
import torch
import json
import random
from tqdm import tqdm
from datasets import load_dataset
import argparse

def read_file(fn):
	with open(fn) as f:
		data = f.read()
	return data

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")

if torch.cuda.is_available():
    device = "cuda"
    print("GPU is available. Using GPU for inference.")
else:
    device = "cpu"
    print("GPU not available. Using CPU for inference.")

model_path="models/OpenHermes-2-Mistral-7B"
input_file="source/wtc_full_reformat.txt"

file_content=read_file(input_file)
chapters=file_content.split("\n\n")
paragraphs=file_content.split("\n")
passage_minlen=300
passage_maxlen=2000
outputfn=input_file.split(".")[0]+"_interview.json"

passages=[]
for chap in chapters:
	passage=""
	for par in chap.split("\n"):
		if(len(passage)<passage_minlen) or not passage[-1]=="." and len(passage)<passage_maxlen:
			passage+="\n" + par
		else:
			passages.append(passage.strip().replace("\n", " "))
			passage=par

prompt_template="""<|im_start|>system
You are an expert dungeon master who interviews a fantasy and science fiction Dungeons and Dragons style story. You formulate situations based on quotes from the story, prompting the actions of the characters, and the environments presented in the story. Formulate a situation that the quote would be the perfect answer to. The situation should be short and directed at creating the geography, interactions between characters, species, and technologies (as well as magic) displayed in the quote. Do not give away the end of the situation. If possible, suggest motivations, events, and perceptions in your situation.

Here is some context that might help you formulate the question regarding the quote:
{ctx}
<|im_end|>
<|im_start|>user
Quote:
{par}<|im_end|>
<|im_start|>assistant
Question:"""

print("Starting processing...")
prompts=[]
for i,p in enumerate(passages):
	#print(f"Processing passage {i+1}/{len(passages)}...")
	if i==0:
		continue
	prompt=prompt_template.format(par=passages[i], ctx=passages[i-1])
	prompts.append(prompt)

prompts_generator=(p for p in prompts)	# pipeline needs a generator, not a list

print(f"{len(chapters)} chapters")
print(f"{len(paragraphs)} paragraphs")
print(f"{len(passages)} passages")

pipeline = transformers.pipeline(
		"text-generation",
		model=model_path,
		torch_dtype=torch.bfloat16,
		device_map="auto",
	)

#pipeline.tokenizer.add_special_tokens({"pad_token":"<pad>"})
#pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))
#pipeline.model.config.pad_token_id = pipeline.tokenizer.pad_token_id

# Check existing special tokens
print("Existing pad token:", pipeline.tokenizer.pad_token)
print("Existing unk token:", pipeline.tokenizer.unk_token)

# Initial tokenizer size
print("Initial tokenizer size:", len(pipeline.tokenizer))

# Add special tokens if they don't exist
#special_tokens_dict = {"pad_token": "<pad>", "unk_token": "<unk>"}
#num_added_toks = pipeline.tokenizer.add_special_tokens(special_tokens_dict)

# Check if resizing is necessary
#if num_added_toks > 0:
#    new_embedding_layer = pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))

# Size after adding tokens
print("Tokenizer size after adding tokens:", len(pipeline.tokenizer))
#	print("Embedding layer size after resizing:", new_embedding_layer.num_embeddings)

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
	"do_sample": True,
	"num_return_sequences": 1,
	"eos_token_id": pipeline.tokenizer.eos_token_id,
	"max_new_tokens": 50,
}

results={
	"model": model_path,
	"input_file": input_file,
	"gen_config": gen_config,
	"passage_minlen": passage_minlen,
	"passage_maxlen": passage_maxlen,
	"num_passages": len(passages),
	"template": prompt_template,
	"interview": []
}

for i, out in enumerate(tqdm(pipeline(prompts_generator, batch_size=2, **gen_config),total=len(prompts))):
	question=out[0]["generated_text"][len(prompts[i]):].strip()
	answer=passages[i+1]

	results["interview"].append({"question": question, "answer": answer})

	write_pretty_json(outputfn,results)
