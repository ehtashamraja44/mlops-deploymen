import pandas as pd
from sklearn.datasets import load_breast_cancer

def ingest():
    disease = load_breast_cancer(as_frame=True).frame
    disease.to_csv("disease.csv", index=False)
    house = pd.read_csv("boston_raw.csv")
    house.to_csv("house.csv", index=False)
    print("Disease:", disease.shape, "| Boston house:", house.shape)

if __name__ == "__main__":
    ingest()
