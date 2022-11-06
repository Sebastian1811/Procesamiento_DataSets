from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
import warnings


warnings.filterwarnings('ignore')
DT_NAME = 'Datasets_procesados/DT_becas-03-11-2022-HORA-16-39-51.csv'
df = pd.read_csv(DT_NAME)

def generateW2vModel():
    """Retorna un modelo w2v usando las keywords del datset como corpus"""
    corpus = []
    for words in df['keywords']: 
        corpus.append(words.split())
    model = Word2Vec(corpus,min_count=1,vector_size=56) 
    return model
    
"""corpus = []
for words in df['keywords']: 
    corpus.append(words.split())

model = Word2Vec(corpus,min_count=1,vector_size=56) 
"""

def vectors(x,model):
    global word_embeddings 
    word_embeddings = []
    for line in df['keywords']:
        avgword2vec = None
        count = 0
        for  word in line.split():
            if word in model.wv.key_to_index :
                count += 1
                if avgword2vec is None:
                    avgword2vec = model.wv[word]
                else:
                    avgword2vec = avgword2vec + model.wv[word]
        if avgword2vec is not None:
            avgword2vec = avgword2vec / count
            word_embeddings.append(avgword2vec)

def recommendations(name):
    #create w2v model
    model = generateW2vModel()
    # Calling the function vectors

    vectors(df,model)
   
    # finding cosine similarity for the vectors

    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings)


    indices = pd.Series(df.index, index = df['name']).drop_duplicates()
         
    idx = indices[name]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    print("puntajes",sim_scores)
    becas_recommendations = [i[0] for i in sim_scores]
    becas_recommendations= np.array(becas_recommendations)
    recommend = df['name'].iloc[becas_recommendations]
    return recommend 

random_row = df.sample(n=1)
random_row = random_row.reset_index(drop=True)  
random_beca_name = random_row['name'][0]

# this example gives full precision "Becas SRE – Universidad Autónoma de Coahuila (UAdeC)"
# another one  "Becas de Máster Universitario en Computación Gráfica, Realidad Virtual y Simulación. Fundación Repsol – Becas Fundación Carolina, 2019"
# one more "Becas Erasmus +, ASTROMUNDUS – Astrophysics, 2018"
print("recomiendame becas parecidas a esta: ",random_beca_name)
print(recommendations(random_beca_name))