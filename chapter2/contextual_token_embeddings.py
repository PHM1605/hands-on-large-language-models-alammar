# DeBERTa v3: best model for producing token embeddings
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# tokens: {input_ids:4-ints,token_type_ids,attention_mask}
tokens = tokenizer("Hello world", return_tensors="pt")

# output: (batch,n_tokens,embed_dim)=(1,4,384)
# model adds 2 tokens at the beginning [CLS] and end [SEP]
output = model(**tokens)[0]

# decode tokens to check
for token in tokens['input_ids'][0]:
  print(tokenizer.decode(token))
