import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import euclidean_distances

DT_NAME ='Datasets_procesados/DT_becas-03-11-2022-HORA-16-39-51.csv'
PORCENTAJEDATASET = 1
RECALCTFIDF = False
DF_SAMPLE = []

df = pd.read_csv(DT_NAME) # Dataset 

def MakeTfIdfMatrix(dataframe=df):
    """Este metodo vectoriza los keywords del dataframe en una matriz tf-idf""" 
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(dataframe['keywords'])
    return tfidf_matrix

def VectorizeRandomSample(dataframe=df):
    """Funcion para vectorizar una muestra aleatoria del dataframe"""
    df_sample = dataframe.sample(frac=PORCENTAJEDATASET) # Extraer muestra aleatoria del dataset 60%
    df_sample = df_sample.reset_index(drop=True)  
    tfidf_matrix = MakeTfIdfMatrix(df_sample) # Vectorización de las palabras clave
    with open("vectorizer.model", 'wb') as fout:
        pickle.dump((tfidf_matrix), fout) # Guardamos el calculo de la vectorización
    return tfidf_matrix,df_sample

def LoadTfidf():
    """Retorna una matriz tfidf previamente calculada desde un file"""
    with open('vectorizer.model', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return tfidf_matrix

def ModelTraining(tfidfMatrix):
    """Retorna un  modelo entrenado usando similaridad coseno"""
    cosine_sim = linear_kernel(tfidfMatrix, tfidfMatrix) # Entranamiento del modelo de recomendación
    return cosine_sim

def get_recommendations(name, df,indices,cosine_sim):
    """Recomendar becas basado en los requisitos"""
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    sim_scores = sim_scores[1:5]
    print("puntajes",sim_scores)
    beca_indices= [i[0] for i in sim_scores]
    beca_indices = np.array(beca_indices)
    #print("beca indices =",beca_indices)
    return df['name'].iloc[beca_indices]

def get_rec_euc(name,df,indices,euc):
    idx = indices[name]
    sim_scores = list(enumerate(euc[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    sim_scores = sim_scores[1:5]
    print("puntajes",sim_scores)
    beca_indices= [i[0] for i in sim_scores]
    beca_indices = np.array(beca_indices)
    #print("beca indices =",beca_indices)
    return df['name'].iloc[beca_indices]

if __name__ == "__main__":
    tfidf_matrix=[]
    random_row = df.sample(n=1)
    random_row = random_row.reset_index(drop=True)  
    random_beca_name = random_row['name'][0]
    if RECALCTFIDF:
        tfidf_matrix,df_sample = VectorizeRandomSample(df)
    else:
        tfidf_matrix = LoadTfidf()
        df_sample = df
    model = ModelTraining(tfidf_matrix)
    indices = pd.Series(df_sample.index, index=df_sample["name"]).drop_duplicates() # type: ignore for ignoring pylance recommendation
    print("recomiendame becas parecidas a esta: ", random_beca_name)  # type: ignore for ignoring pylance recommendation
    print(get_recommendations(df_sample['name'][3],df_sample,indices,model)) # type: ignore for ignoring pylance recommendation
    print("**********************")
    euc = euclidian = euclidean_distances(tfidf_matrix, tfidf_matrix)
    print("recomiendame becas parecidas a esta: ", random_beca_name)
    print(get_rec_euc(df_sample['name'][3],df_sample,indices,euc))
