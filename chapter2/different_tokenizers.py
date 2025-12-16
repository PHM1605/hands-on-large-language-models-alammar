from transformers import AutoTokenizer
import tiktoken

text = """
English and CAPITALIZATION
🎵鸟
show_tokens False None elif == >= else: two tabs:"  " Three tabs: "    "
12.0*50=600
"""
colors_list = [
  '102;194;165', '252;141;98', '141;160;203',
  '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence, tokenizer_name):
  if tokenizer_name == "gpt4":
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_ids = tokenizer.encode(sentence)
  else:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
  for idx, t in enumerate(token_ids):
    print(f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' + tokenizer.decode([t]) + '\x1b[0m', end=' ')
  print("\n")

# show_tokens(text, tokenizer_name='bert-base-uncased')
# show_tokens(text, tokenizer_name='bert-base-cased')
# show_tokens(text, tokenizer_name='gpt2')
# show_tokens(text, tokenizer_name='google/flan-t5-small')
# show_tokens(text, tokenizer_name='gpt4')
# show_tokens(text, tokenizer_name='bigcode/starcoder2-3b')
# show_tokens(text, tokenizer_name="facebook/galactica-125m")
show_tokens(text, tokenizer_name="microsoft/phi-3-mini-4k-instruct")