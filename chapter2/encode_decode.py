from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
  "microsoft/Phi-3-mini-4k-instruct",
  device_map="cuda:0",
  torch_dtype="auto",
  trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
  "microsoft/Phi-3-mini-4k-instruct"
)
prompt = "Write an email appologizing to Sarah for the tragic gardening mishap. Explain how it happend.<|assistant|>"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
generation_output = model.generate(
  input_ids=input_ids,
  max_new_tokens=20
)
print(tokenizer.decode(generation_output[0]))
print(input_ids)

# Inspect every tokens
for id in input_ids[0]:
  print(tokenizer.decode(id))

# Generated output (ids)
print(generation_output[0])

# Decode some words
print("One word decode:", tokenizer.decode(3323))
print("One word decode:", tokenizer.decode(622))
print("One word decode:", tokenizer.decode(29901))
print("One word decode:", tokenizer.decode(317))
