from datasets import load_dataset
from transformers import pipeline
import numpy as np 
from tqdm import tqdm 
from transformers.pipelines.pt_utils import KeyDataset 

# Load movie reviews data
data = load_dataset("rotten_tomatoes")
print("Loaded dataset:\n", data)
print("One sample:\n", data["train"][0,-1])

# Load model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
pipe = pipeline(
  model=model_path,
  tokenizer=model_path,
  top_k=None, 
  device="cuda:0"
)

# Run model inference
y_pred = []
for output in tqdm(
  pipe(KeyDataset(data["test"], "text")), # take a string i.e. one <review> then passed through <pipe>
  total=len(data["test"])
):
  # [{label:"negative",score:xx},{label:"neutral",score:yy},{label:"positive",score:zz}]
  negative_score = output[0]["score"]
  positive_score = output[2]["score"]
  assignment = np.argmax([negative_score, positive_score])
  y_pred.append(assignment)

## Evaluation
from sklearn.metrics import classification_report 

def evaluate_performance(y_true, y_pred):
  performance = classification_report(
    y_true, y_pred, # each is list of 0 or 1
    target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

evaluate_performance(data["test"]["label"], y_pred)
