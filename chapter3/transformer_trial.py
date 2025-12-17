import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
  "microsoft/Phi-3-mini-4k-instruct",
  device_map="cuda",
  torch_dtype="auto",
  trust_remote_code=True 
)
generator = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  return_full_text=False,
  max_new_tokens=50,
  do_sample=False 
)
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
output = generator(prompt)
print(output[0]['generated_text'])
print(model)

# Try 1 output prompt
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids # (1,5)
input_ids = input_ids.to("cuda")
model_output = model.model(input_ids) # list of 1 element, which is (1,5,3072)
lm_head_output = model.lm_head(model_output[0]) # (1,5,32064) with vocab_size=32064
# get the generated word
token_id = lm_head_output[0,-1].argmax(-1)
print("Generated word: ", tokenizer.decode(token_id))