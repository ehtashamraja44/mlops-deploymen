from sklearn.datasets import load_breast_cancer, fetch_california_housing

def ingest():
    disease = load_breast_cancer(as_frame=True).frame
    disease.to_csv("disease.csv", index=False)
    house = fetch_california_housing(as_frame=True).frame
    house = house.sample(5000, random_state=42)
    house.to_csv("house.csv", index=False)
    print("Disease:", disease.shape, "| House:", house.shape)

if __name__ == "__main__":
    ingest()
