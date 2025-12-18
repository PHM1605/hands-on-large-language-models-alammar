import openai, os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
OPENAI_KEY=os.getenv("OPENAI_KEY")
client = openai.OpenAI(api_key=OPENAI_KEY)

def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
  ]
  chat_completion = client.chat.completions.create(
    messages=messages,
    model=model,
    temperature=0,
    max_tokens=1,
    stop=["\n", " "] # stop immediately
  )
  return chat_completion.choices[0].message.content.strip()

prompt = """
  Predict whether the following document is a positive or negative movie review:

  [DOCUMENT]

  If it is positive return 1 and if it is negative return 0.
"""

# # Test one call
# document = "unpretentious, charming, quirky, original"
# predictions =chatgpt_generation(prompt, document)

# Test many calls
from datasets import load_dataset
# data: {train,validation,test}
# each, e.g. <train> is {'text':[],'label':[]}
data = load_dataset("rotten_tomatoes")
predictions = [
  chatgpt_generation(prompt, doc) for doc in tqdm(data["test"]["text"])
]

def to_int(x):
  return 1 if x=="1" else 0

y_pred = [to_int(pred) for pred in predictions]

## Evaluation
from sklearn.metrics import classification_report 

def evaluate_performance(y_true, y_pred):
  performance = classification_report(
    y_true, y_pred, # each is list of 0 or 1
    target_names=["Negative Review", "Positive Review"]
  )
  print(performance)
evaluate_performance(data["test"]["label"], y_pred)