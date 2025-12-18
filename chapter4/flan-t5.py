from transformers import pipeline 
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from tqdm import tqdm 

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
# data: {'text':['abc','xz',...],'label':[0,1,...], "t5":["xx","yy",...]}
data = data.map(lambda example: {"t5": prompt+example["text"]})
print("Data with <t5> column for prompt:\n", data)

# run inference 
y_pred = [] 
# output: [{generated_text:"negative"}]
for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
  text = output[0]["generated_text"]
  y_pred.append(0 if text=="negative" else 1)

# evaluate 
from sklearn.metrics import classification_report 

def evaluate_performance(y_true, y_pred):
  performance = classification_report(
    y_true, y_pred, # each is list of 0 or 1
    target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

evaluate_performance(data["test"]["label"], y_pred)