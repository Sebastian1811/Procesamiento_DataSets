import spacy
from collections import Counter
from string import punctuation
from string import digits
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pickle


nlp = spacy.load("es_dep_news_trf") #Modelo entrenado para el procesamiento de los requisitos
df = pd.read_csv('Datasets_procesados/DT_becas_noNans_noIndex.csv') # Dataset 


"""
Este metodo extrae las palabras claves del documento (en nuestro caso la columna requisitos)
procesa los requisitos usando el pipelines de  spacy
"""
def get_hotwords(text):
    result = []
    pos_tag = ['NOUN','VERB'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation or token.text in digits):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

"""
Este metodo vectoriza las palabras clave que se extraigan a partir del pipeline de spacy
"""
def MakeTfIdfMatrix(dataframe):

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(dataframe['keywords'])
    #print(len(tfidf.vocabulary_))
    return tfidf_matrix
"""
Este metodo procesa el texto de los requisitos y extrae las palabras clave.
Aplica steaming, lemmatizer, tagging, etc
"""
def RequirementsPipelineProcessing(dataframe):
    keywords = []
    for i in range(len(dataframe.index)):
        #output.update(get_hotwords(requisitos[i]))
        output = set(get_hotwords(requisitos[i]))
        most_common_list = Counter(output).most_common(20)
        lista_keywords = list(map(lambda x: x[0],most_common_list))
        keywords_string = ' '.join(lista_keywords)
        keywords.append(keywords_string)
    return keywords

"""
Funcion para vectorizar una muestra aleatoria del dataframe
"""
def VectorizeRandomSample(dataframe):
    df_sample = dataframe.sample(frac=0.60) # Extraer muestra aleatoria del dataset 60%
    keywords = RequirementsPipelineProcessing(df_sample) # Extraccion palabras claves de los requisitos 
    df_sample['keywords'] = keywords # Inserción de las palabras al dataset
    tfidf_matrix = MakeTfIdfMatrix(df_sample) # Vectorización de las palabras clave
    with open("vectorizer.model", 'wb') as fout:
        pickle.dump((tfidf_matrix), fout) # Guardamos el calculo de la vectorización
    return tfidf_matrix, df_sample

"""
Entrenammiento del modelo usando similaridad coseno
"""
def ModelTraining(tfidfMatrix):
    cosine_sim = linear_kernel(tfidfMatrix, tfidfMatrix) # Entranamiento del modelo de recomendación
    return cosine_sim


df_replic = df.sample(frac=0.60)
df_replic = df_replic.reset_index(drop=True)


requisitos = df_replic['requirements']

    
#output = set(get_hotwords(text))
#output= set()
keywords = []
for i in range(len(df_replic.index)):
    #output.update(get_hotwords(requisitos[i]))
    output = set(get_hotwords(requisitos[i]))
    most_common_list = Counter(output).most_common(20)
    lista_keywords = list(map(lambda x: x[0],most_common_list))
    keywords_string = ' '.join(lista_keywords)
    keywords.append(keywords_string)


df_replic['keywords'] = keywords

tfidf_matrix = MakeTfIdfMatrix(df_replic)

print(df_replic['keywords'])
print(tfidf_matrix)




#TO DO APLICAR KEYWORDS A TODAS LAS FILAS DEL DT
#HACERLOS SOBRE UNA COPIA DEL DT
# GUARDAR EL NUEVO DT Y APLICAR TF-IDF

#print(output)
#most_common_list = Counter(output).most_common(30)

"""for item in most_common_list:
  print(item[0])"""

  #QUE HAGO CON  LOS PAISES
  # con los niveles de estudio
from sklearn.metrics.pairwise import cosine_similarity


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df_replic.index, index=df_replic['name']).drop_duplicates()

def get_recommendations(name, cosine_sim=cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    sim_scores = sim_scores[1:5]
    print("puntajes",sim_scores)
    beca_indices= [i[0] for i in sim_scores]
    beca_indices = np.array(beca_indices)
    #print("beca indices =",beca_indices)

    return df_replic['name'].iloc[beca_indices]

#print(get_recommendations("Becas OEA – Colorado State University, 2022"))
#print("recomiendame becas parecidas a:",df_replic['name'][3] )
#print(get_recommendations(df_replic['name'][3]))

def get_recommendations2(name, df,indices,cosine_sim=cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    sim_scores = sim_scores[1:5]
    print("puntajes",sim_scores)
    beca_indices= [i[0] for i in sim_scores]
    beca_indices = np.array(beca_indices)
    #print("beca indices =",beca_indices)

    return df['name'].iloc[beca_indices]

print("code refactor")
vector,df2 = VectorizeRandomSample(df)
model = ModelTraining(vector)
indices = pd.Series(df2.index, index=df2['name']).drop_duplicates()
get_recommendations2(df2['name'][3],df2,indices,model)
#print(cosine_similarity(tfidf_matrix[0:1],tfidf_matrix).flatten())