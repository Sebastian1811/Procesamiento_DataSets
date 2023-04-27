import pandas as pd 
from datetime import datetime

DT = 'Datasets_Origen/DT2-SinFronteras-SemiFinal.csv'

def cleanDF():
    df = pd.read_csv(DT)
    df.dropna(inplace = True)
   # df.to_csv('becasRequisitos3.csv', index=False)
    df.to_csv(f'otros_DTS/DT-becas-prepro-{TIMESTAMP}.csv', index=False)

if __name__ == "__main__":
    TIMESTAMP = datetime.now()
    TIMESTAMP = TIMESTAMP.strftime("%d-%m-%Y-HORA-%H-%M-%S")
    cleanDF(TIMESTAMP)
