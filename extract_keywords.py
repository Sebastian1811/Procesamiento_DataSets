import spacy
import pandas as pd
from collections import Counter
from string import punctuation
from string import digits
from datetime import datetime

nlp = spacy.load("es_dep_news_trf") #Modelo entrenado para el procesamiento de los requisitos
df = pd.read_csv('Datasets_procesados/DT_becas_noNans_noIndex.csv') # Dataset 

def get_hotwords(text):
    """
    Este metodo extrae las palabras claves del documento (en nuestro caso la columna requisitos)
    procesa los requisitos usando el pipelines de spacy. Retorna una lista con las palabras claves.
    Aplica steaming, lemmatizer, tagging, etc.
    """
    result = []
    pos_tag = ['NOUN','VERB'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation or token.text in digits):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

def RequirementsPipelineProcessing(dataframe):
    """
    Este metodo procesa el texto de los requisitos y extrae las palabras clave de
    todas las filas del dataset.
    """
    keywords = []
    requisitos = dataframe['requirements']
    for i in range(len(dataframe.index)):
        #output.update(get_hotwords(requisitos[i]))
        output = set(get_hotwords(requisitos[i]))
        most_common_list = Counter(output).most_common(20)
        lista_keywords = list(map(lambda x: x[0],most_common_list))
        keywords_string = ' '.join(lista_keywords)
        keywords.append(keywords_string)
    return keywords

if __name__ == "__main__":
    TIMESTAMP = datetime.now()
    TIMESTAMP = TIMESTAMP.strftime("%d-%m-%Y-HORA-%H-%M-%S")
    keywords = RequirementsPipelineProcessing(df)
    df['keywords'] = keywords
    df.to_csv(f'Datasets_procesados/DT_becas-{TIMESTAMP}.csv',index=False)

#TO DO APLICAR KEYWORDS A TODAS LAS FILAS DEL DT
#HACERLOS SOBRE UNA COPIA DEL DT

#QUE HAGO CON  LOS PAISES
# con los niveles de estudio