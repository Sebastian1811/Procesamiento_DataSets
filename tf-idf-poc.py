import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def GetDtFromCSV():
    df = pd.read_csv('prueba.csv')
    return df

def MakeTfIdfMatrix(df):
    #df = GetDtFromCSV()
    v = TfidfVectorizer()
    x = v.fit_transform(df['keywords'])
    #a = x.toarray()
    #s = np.shape(x)
    """"print(x)
    mat = np.matrix(a)
    with open('outfile.txt','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.4f')"""
    return x
    

if __name__ == "__main__":
    df = GetDtFromCSV()
    MakeTfIdfMatrix(df)
