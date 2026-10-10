import sys, mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
src_alias, dst_alias = sys.argv[1], sys.argv[2]
client = MlflowClient()
for name in ["DiseaseClassifier", "HousePriceRegressor"]:
    mv = client.get_model_version_by_alias(name, src_alias)
    client.set_registered_model_alias(name, dst_alias, mv.version)
    print(f"{name}: v{mv.version} {src_alias} -> {dst_alias}")
