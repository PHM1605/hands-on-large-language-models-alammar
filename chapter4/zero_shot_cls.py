# Unsupervised classification method
# Zero-shot classification
from sentence_transformers import SentenceTransformer
from datasets import load_dataset 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

data = load_dataset("rotten_tomatoes")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

# create embeddings for our labels; (2,embed_dim)
label_embeddings = model.encode(["A negative review", "A positive review"])
print("Label embeddings:\n", label_embeddings)

def evaluate_performance(y_true, y_pred):
  performance = classification_report(
    y_true, y_pred, # each is list of 0 or 1
    target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

# how close the <document-embed> to the 2 <label-embed>s?
sim_matrix = cosine_similarity(test_embeddings, label_embeddings) # (batch,2)
y_pred = np.argmax(sim_matrix, axis=1)
evaluate_performance(data["test"]["label"], y_pred)