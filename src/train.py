import sys, pandas as pd, mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

alias = sys.argv[1] if len(sys.argv) > 1 else "Challenger"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

def run_experiment(exp_name, model_name, csv, target, models, task):
    mlflow.set_experiment(exp_name)
    df = pd.read_csv(csv)
    X, y = df.drop(target, axis=1), df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    best_score, best_run, best_name = -1e9, None, None
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            if task == "classification":
                score = accuracy_score(y_te, pred)
                mlflow.log_metric("accuracy", score)
            else:
                score = r2_score(y_te, pred)
                mlflow.log_metric("r2", score)
                mlflow.log_metric("rmse", mean_squared_error(y_te, pred) ** 0.5)
            mlflow.log_param("model_type", name)
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
            if score > best_score:
                best_score, best_run, best_name = score, run.info.run_id, name
    result = mlflow.register_model(f"runs:/{best_run}/model", model_name)
    client.set_registered_model_alias(model_name, alias, result.version)
    print(f"[{model_name}] best={best_name} v{result.version} alias={alias} score={best_score:.3f}")

run_experiment("disease-classification", "DiseaseClassifier", "disease.csv", "target",
    {"RandomForest": RandomForestClassifier(n_estimators=100),
     "LogisticRegression": LogisticRegression(max_iter=5000)}, "classification")
run_experiment("house-price-regression", "HousePriceRegressor", "house.csv", "MedHouseVal",
    {"RandomForestRegressor": RandomForestRegressor(n_estimators=100)}, "regression")
