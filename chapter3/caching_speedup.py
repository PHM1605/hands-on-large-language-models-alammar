import time 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
  "microsoft/Phi-3-mini-4k-instruct",
  device_map="cuda",
  torch_dtype="auto",
  trust_remote_code=True 
)

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

# Warm-up
with torch.no_grad():
  model.generate(input_ids=input_ids, max_new_tokens=10)
torch.cuda.synchronize()

# When cache True
start =time.perf_counter()
with torch.no_grad():
  generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    use_cache=True   
  )
torch.cuda.synchronize()
time_cache = time.perf_counter() - start

# When cache False
start = time.perf_counter()
with torch.no_grad():
  generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    use_cache=False
  )
torch.cuda.synchronize()
time_no_cache = time.perf_counter() - start 

print(f"use_cache=True: {time_cache:.3f} s")
print(f"use_cache=False: {time_no_cache:.3f} s")