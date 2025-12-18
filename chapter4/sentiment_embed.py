# use and EMBEDDING MODEL to extract features
# then passing features through a CLASSIFIER
from datasets import load_dataset 
from sentence_transformers import SentenceTransformer 

data = load_dataset("rotten_tomatoes")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
print("Train embeddings shape:\n", train_embeddings.shape) # (num_docs,embed_dim)

# classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

# evaluation
from sklearn.metrics import classification_report 

def evaluate_performance(y_true, y_pred):
  performance = classification_report(
    y_true, y_pred, # each is list of 0 or 1
    target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)