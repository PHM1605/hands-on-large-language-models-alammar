import pandas as pd 
from urllib import request 

data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')
# skip the first 2 lines (metadata)
lines = data.read().decode("utf-8").split("\n")[2:]
# remove playlists with only one song
# line: '23 1 15...'
# => playlists: list of [23,1,15...] each
playlists = [line.rstrip().split() for line in lines if len(line.split())>1] 
print('Playlist #1:\n ', playlists[0], '\n')
print('Playlist #2:\n ', playlists[1])

# Load song metadata
songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
# each line: <id>\t<title>\t<artist>
songs_file = songs_file.read().decode("utf-8").split("\n")
# songs: [[<id>,<title>,<artist>],[],]
songs = [s.rstrip().split('\t') for s in songs_file]

songs_df = pd.DataFrame(data=songs, columns=['id','title','artist'])
songs_df = songs_df.set_index('id')

# train our model
from gensim.models import Word2Vec 
model = Word2Vec(
  playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4
)

# ask model to choose songs similar to song 2172
song_id = 2172
print("Root song:\n", songs_df.iloc[2172])
print("Similar result:\n", model.wv.most_similar(positive=str(song_id))) # [('3167', 0.99), ('2849', 0.97),]

# extract recommendations
import numpy as np
def print_recommendations(song_id):
  # array: [['3167', '0.99'], ['2849', '0.97'],]
  # similar_songs: ['3167','2849',]
  similar_songs = np.array(
    model.wv.most_similar(positive=str(song_id), topn=5)
  )[:,0]
  return songs_df.iloc[similar_songs]

print("Recommendations:\n", print_recommendations(2172))