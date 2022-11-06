import pandas as pd
import numpy as np
import nltk


from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from matplotlib import pyplot
from gensim.models import KeyedVectors

DT_NAME = 'Datasets_procesados/DT_becas-03-11-2022-HORA-16-39-51.csv'
df = pd.read_csv(DT_NAME)

corpus = []
for words in df['keywords']: #type: ignore
    corpus.append(words.split())












EMBEDDING_FILE = 'C:/Users/eljho/Downloads/GoogleNews-vectors-negative300.bin'
#path= 'C:/Users/eljho/Downloads/GoogleNews-vectors-negative300.bin'
#model = KeyedVectors.load_word2vec_format(path, binary=True, limit=20000) ESTO FUNCIONO


google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True,limit=20000)

# Training our corpus with Google Pretrained Model

google_model = Word2Vec(vector_size = 300, window=5, min_count = 2, workers = -1) #type: ignore
google_model.build_vocab(corpus)

#model.intersect_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', lockf=1.0, binary=True)


#google_model.intersect_word2vec_format(EMBEDDING_FILE, lockf=1.0, binary=True) 

google_model.train(corpus, total_examples=google_model.corpus_count, epochs = 5)


def vectors(x):
    
    # Creating a list for storing the vectors (description into vectors)
    global word_embeddings
    word_embeddings = []

    # Reading the each book description 
    for line in df['keywords']:
        avgword2vec = None
        count = 0
        for word in line.split():
            if word in google_model.wv.key_to_index:
                count += 1
                if avgword2vec is None:
                    avgword2vec = google_model.wv[word]
                else:
                    avgword2vec = avgword2vec + google_model.wv[word]
                
        if avgword2vec is not None:
            avgword2vec = avgword2vec / count
        
            word_embeddings.append(avgword2vec)

def recommendations(title):
    
    # Calling the function vectors

    vectors(df)
    
    # finding cosine similarity for the vectors

    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings)

    # taking the title and book image link and store in new data frame called becas
    becas = df[['name', 'url']]
    #Reverse mapping of the index
    indices = pd.Series(df.index, index = df['name']).drop_duplicates()
         
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    recommend = becas.iloc[book_indices]
    return recommend
  
random_row = df.sample(n=1)
random_row = random_row.reset_index(drop=True)  
random_beca_name = random_row['name'][0]
print("recomiendame becas parecidas a esta: ",random_beca_name)
print(recommendations(random_beca_name))