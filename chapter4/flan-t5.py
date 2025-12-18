from transformers import pipeline 
from datasets import load_dataset

# data: {train,validation,test}
# each, e.g. <train> is {'text':[],'label':[]}
data = load_dataset("rotten_tomatoes")
pipe = pipeline(
  "text2text-generation",
  model="google/flan-t5-small",
  device="cuda:0"
)
prompt = "Is the following sentence positive or negative? "
# example: {'text':['abc','xz',...],'label':[0,1,...]}
data = data.map(lambda example: {"t5": prompt+example["text"]})
print("Data with <t5> column for prompt:\n", data)