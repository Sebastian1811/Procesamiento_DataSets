import spacy
from collections import Counter
from string import punctuation
from string import digits
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
nlp = spacy.load("es_dep_news_trf")


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
def MakeTfIdfMatrix(df):

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['keywords'])
    print(len(tfidf.vocabulary_))
    return tfidf_matrix

df = pd.read_csv('Datasets_procesados/DT_becas_noNans_noIndex.csv')
#df = pd.read_csv('Datasets_procesados/DT_becas_limpio_index.csv')


df_replic = df.sample(frac=0.20)
df_replic = df.reset_index(drop=True)

#print(len(df_replic.index))
#print(df_replic.shape)
#print(df_replic.head())


requisitos = df_replic['requirements']

    
#output = set(get_hotwords(text))
#output= set()
keywords = []
for i in range(len(df_replic.index)):
    #output.update(get_hotwords(requisitos[i]))
    output = set(get_hotwords(requisitos[i]))
    most_common_list = Counter(output).most_common(3)
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
    print("beca indices =",beca_indices)

    return df_replic['name'].iloc[beca_indices]

print(get_recommendations("Becas OEA – Colorado State University, 2022"))

#print(cosine_similarity(tfidf_matrix[0:1],tfidf_matrix).flatten())