
import pickle
import pandas as pd
import numpy as np


def loadW2v():
    """Retorna una matriz tfidf previamente calculada desde un file"""
    with open('w2v.model', 'rb') as f:
        w2v = pickle.load(f)
    return w2v

mat = loadW2v() 
DT_NAME = 'Datasets_procesados/DT_becas-03-11-2022-HORA-16-39-51.csv'
df = pd.read_csv(DT_NAME)

def recommendations(name):
    indices = pd.Series(df.index, index = df['name']).drop_duplicates()
    idx = indices[name]
    return mat[idx],idx

def calcPearsonCorrelation(myvector,idx):
    y = []
    sim_scores = []
    for i in range(len(mat)):
        if i <= len(mat)-2 and i != idx:
            #print(i)
            y = np.corrcoef(myvector, mat[i])
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

random_row = df.sample(n=1)
random_row = random_row.reset_index(drop=True)  
random_beca_name = random_row['name'][0]
vector,idx= recommendations(random_beca_name)
print("recomiendame becas parecidas a esta: ",random_beca_name)
print(calcPearsonCorrelation(vector,idx))
     

