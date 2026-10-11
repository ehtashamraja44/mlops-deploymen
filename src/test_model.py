import sys, pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
alias = sys.argv[1] if len(sys.argv) > 1 else "Challenger"

def check(model_name, csv, target, metric, threshold):
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    df = pd.read_csv(csv)
    X, y = df.drop(target, axis=1), df[target]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    score = metric(y_te, model.predict(X_te))
    print(f"{model_name}@{alias}: {score:.3f} (min {threshold})")
    return score >= threshold

ok = check("DiseaseClassifier", "disease.csv", "target", accuracy_score, 0.90)
ok = check("HousePriceRegressor", "house.csv", "MEDV", r2_score, 0.60) and ok
sys.exit(0 if ok else 1)
