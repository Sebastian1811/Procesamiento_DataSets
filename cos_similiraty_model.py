import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pickle


df = pd.read_csv('Datasets_procesados/DT_becas-03-11-2022-HORA-16-39-51.csv') # Dataset 
PORCENTAJEDATASET = 0.60
RECALCTFIDF=True
DF_SAMPLE=[]

"""
Este metodo vectoriza la columna keywords del dataframe
"""
def MakeTfIdfMatrix(dataframe=df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(dataframe['keywords'])
    #print(len(tfidf.vocabulary_))
    return tfidf_matrix

"""
Funcion para vectorizar una muestra aleatoria del dataframe
"""
def VectorizeRandomSample(dataframe=df):
    df_sample = dataframe.sample(frac=PORCENTAJEDATASET) # Extraer muestra aleatoria del dataset 60%
    df_sample = df_sample.reset_index(drop=True)  
    tfidf_matrix = MakeTfIdfMatrix(df_sample) # Vectorización de las palabras clave
    with open("vectorizer.model", 'wb') as fout:
        pickle.dump((tfidf_matrix), fout) # Guardamos el calculo de la vectorización
    return tfidf_matrix,df_sample

"""
Funcion para cargar una matriz tfidf previamente calculada
"""
def LoadTfidf():
    with open('vectorizer.model', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return tfidf_matrix

"""
Entrenammiento del modelo usando similaridad coseno
"""
def ModelTraining(tfidfMatrix):
    cosine_sim = linear_kernel(tfidfMatrix, tfidfMatrix) # Entranamiento del modelo de recomendación
    return cosine_sim
"""
Recomendar becas basado en los requisitos
"""
def get_recommendations(name, df,indices,cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    sim_scores = sim_scores[1:5]
    print("puntajes",sim_scores)
    beca_indices= [i[0] for i in sim_scores]
    beca_indices = np.array(beca_indices)
    #print("beca indices =",beca_indices)
    return df['name'].iloc[beca_indices]

if __name__ == "__main__":
    tfidf_matrix=[]
    if RECALCTFIDF:
        tfidf_matrix,df_sample = VectorizeRandomSample(df)
    else:
        tfidf_matrix = LoadTfidf()
        df_sample = DF_SAMPLE

    model = ModelTraining(tfidf_matrix)
    indices = pd.Series(df_sample.index, index=df_sample["name"]).drop_duplicates() # type: ignore for ignoring pylance recommendation
    
    print("recomiendame becas parecidas a esta: ", df_sample['name'][3])  # type: ignore for ignoring pylance recommendation
    print(get_recommendations(df_sample['name'][3],df_sample,indices,model)) # type: ignore for ignoring pylance recommendation