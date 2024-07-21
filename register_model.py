# client demo

from mlflow.tracking import MlflowClient
import mlflow
# Initialize the MLflow Client
client = MlflowClient()

# Replace with the run_id of the run where the model was logged
run_id = "ee2671fb022c47f89c27c9aa3ef25f42"

# Replace with the path to the logged model within the run
model_path = "file:///C:/Users/Nitish/Desktop/mlops/model-registry-demo/mlruns/406765814162474516/ee2671fb022c47f89c27c9aa3ef25f42/artifacts/random_forest"

# Construct the model URI
model_uri = f"runs:/{run_id}/{model_path}"

# Register the model in the model registry
model_name = "diabetes-rf"
result = mlflow.register_model(model_uri, model_name)

import time
time.sleep(5)

# Add a description to the registered model version
client.update_model_version(
    name=model_name,
    version=result.version,
    description="This is a RandomForest model trained to predict diabetes outcomes based on Pima Indians Diabetes Dataset."
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="experiment",
    value="diabetes prediction"
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="day",
    value="sat"
)
print(f"Model registered with name: {model_name} and version: {result.version}")
print(f"Added tags to model {model_name} version {result.version}")

# Get and print the registered model information
registered_model = client.get_registered_model(model_name)
print("Registered Model Information:")
print(f"Name: {registered_model.name}")
print(f"Creation Timestamp: {registered_model.creation_timestamp}")
print(f"Last Updated Timestamp: {registered_model.last_updated_timestamp}")
print(f"Description: {registered_model.description}")

