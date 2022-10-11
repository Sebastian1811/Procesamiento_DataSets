import pandas as pd 


def cleanDF():
    df = pd.read_csv('Datasets_Origen/DT2-SinFronteras-SemiFinal.csv')

    df.dropna(inplace = True)
    df.to_csv('becasRequisitos3.csv', index=False)

if __name__ == "__main__":
    cleanDF()
