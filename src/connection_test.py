import dagshub, mlflow

dagshub.init(repo_owner="shyamashreeghorai24", repo_name="MLOPS-MLFlOW", mlflow=True)

mlflow.set_experiment("connection_test")

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.99)

print("✅ Successfully logged a test run to DagsHub!")
