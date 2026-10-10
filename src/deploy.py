import sys, mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
alias = sys.argv[1] if len(sys.argv) > 1 else "Challenger"
for name in ["DiseaseClassifier", "HousePriceRegressor"]:
    mv = MlflowClient().get_model_version_by_alias(name, alias)
    print(f"Deploying {name} version {mv.version} (alias {alias})")
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@{alias}", dst_path=f"deployed/{name}")
print("Deploy done")
