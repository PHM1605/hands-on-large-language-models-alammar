from datasets import load_dataset 

dataset = load_dataset("maartengr/arxiv_nlp")["train"]
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# select am embedding generating model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
print("Embedding shape: ", embeddings.shape) # (n_abstracts,embed_dim)=(44949,384)

# dimension reduction with UMAP
from umap import UMAP
umap_model = UMAP(
  n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)
print("Dimension-reduced shape: ", reduced_embeddings.shape) # (n_abstracts,reduced_dim)=(44949,5)

# clustering
from hdbscan import HDBSCAN 
hdbscan_model = HDBSCAN(
  
)