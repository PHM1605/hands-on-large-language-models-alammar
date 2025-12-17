import gensim.downloader as api 

model = api.load("glove-wiki-gigaword-50")
# find the most similar words to "king"
print(model.most_similar([model["king"]], topn=11))
