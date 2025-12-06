"""Prediccion script for the MLflow model.

This script loads a trained model from MLflow and makes predictions

$ python3 make_predicitions.py
"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
X = df.drop("quality", axis=1)

## Debe verificarse el run_id del modelo que se quiere cargar
## Se puede obtener el run_id desde la interfaz de MLflow

logged_model = "runs:/e8183756a430413b922bfb1dbe32210b/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(X)

print(y)
