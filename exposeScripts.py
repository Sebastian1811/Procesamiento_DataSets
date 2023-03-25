from w2v import recommendations as tfidf
from w2v import rec_euc as euclidean
import pandas as pd

def get_tfid_recommendations(beca_name):
    recommendations = tfidf(beca_name) 
    print(recommendations)




get_tfid_recommendations("Becas SRE – Universidad Autónoma de Coahuila (UAdeC)")