from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')

DT_NAME = 'Datasets_procesados/DT_becas-03-11-2022-HORA-16-39-51.csv'
RECALC_MODEL = False

df = pd.read_csv(DT_NAME)

def generateW2vModel():
    """Retorna un modelo w2v entrenado usando un porcentaje
     de las keywords del datset como corpus"""
    corpus = []
    for words in df['keywords']: 
        corpus.append(words.split())
    corpus = np.array(corpus)
    corpus, corpus30 = np.split(corpus, [int(0.6 * len(corpus))])
    corpus = corpus.tolist()
    model = Word2Vec(vector_size = 300, window=5, min_count = 2, workers = -1) 
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs = 5)
    return model

def vectors(x,model):
    """Vectoriza todas las keywords de los requisitos 
    del dataframe utilizando los scores del modelo entrenado"""
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
    saveVectors(word_embeddings)

def recommendations(name):
    """Genera un listado de recomendaciones de becas basado en los vectores 
    de los keywords de los requisitos usando medida de similitud coseno"""             
    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings) # finding cosine similarity for the vectors
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

def recommendationsScores(name):
    """Genera un listado con los puntajes de similitud de todas las becas.
    Método desarrollado para testear el motor"""             
    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings) # finding cosine similarity for the vectors
    indices = pd.Series(df.index, index = df['name']).drop_duplicates()
    idx = indices[name]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    for i in sim_scores:
        print(i[1],',',end='')
    print("\n")

def saveVectors(vectors):
     """Almacenar calculo de los vectores de las keywords en un archivo"""
     with open("w2v.model", 'wb') as fout:
        pickle.dump((vectors), fout) # Guardamos el calculo de la vectorización

def loadVectors():
    """Leer calculo de los vectores de las keywords en un archivo"""
    with open('w2v.model', 'rb') as f:
        w2vModel = pickle.load(f)
    return w2vModel

def  rec_euc(name):
    """Genera un listado de recomendaciones de becas
    basado en los vectores de los requisitos usando la métrica 
    de la distancia euclidiana"""
    euclidian = euclidean_distances(word_embeddings, word_embeddings) # finding euclidean distance for the vectors
    indices = pd.Series(df.index, index = df['name']).drop_duplicates()
    idx = indices[name]
    sim_scores = list(enumerate(euclidian[idx])) #type: ignore
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = False)
    sim_scores = sim_scores[1:6]
    print("puntajes",sim_scores)
    becas_recommendations = [i[0] for i in sim_scores]
    becas_recommendations= np.array(becas_recommendations)
    recommend = df['name'].iloc[becas_recommendations]
    return recommend 

def getVectorForPearson(name):
    """Método para garantizar funcionamiento de las recomendaciones
    usando correlación de pearson"""
    indices = pd.Series(df.index, index = df['name']).drop_duplicates()
    idx = indices[name]
    return word_embeddings[idx],idx

def pearsonRecommendations(myvector,idx):
    """Genera un listado de recomendaciones de becas
    basado en los vectores de los requisitos usando la correlación
    de Pearson"""
    y = []
    sim_scores = []
    for i in range(len(word_embeddings)):
        if i <= len(word_embeddings)-2 and i != idx:
            #print(i)
            y = np.corrcoef(myvector, word_embeddings[i])
            y = y[0][1]
            sim_scores.append(list([i,y]))
            #print(y)
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    print(sim_scores)
    becas_recommendations = [i[0] for i in sim_scores]
    becas_recommendations= np.array(becas_recommendations)
    recommend = df['name'].iloc[becas_recommendations]
    return recommend

if __name__ == "__main__":
    random_row = df.sample(n=1)
    random_row = random_row.reset_index(drop=True)  
    random_beca_name = random_row['name'][0]

    if RECALC_MODEL:
        model = generateW2vModel()
        vectors(df,model)
    else:
        word_embeddings=loadVectors()

    # for i in df['name']:
    #     recommendationsScores(i)
        
    print("************************ COSINE SIMILARITY ************************************")
    print("recomiendame becas parecidas a esta: ",random_beca_name)
    print(recommendations(random_beca_name))
    print("************************ EUCLIDEAN DISTANCE ************************************")
    print("recomiendame becas parecidas a esta: ",random_beca_name)
    print(rec_euc(random_beca_name))
    print("************************ PEARSON CORRELATION ************************************")
    print("recomiendame becas parecidas a esta: ",random_beca_name)
    vector,idx = getVectorForPearson(random_beca_name)
    print(pearsonRecommendations(vector,idx))