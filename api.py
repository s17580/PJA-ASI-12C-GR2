from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI()

# Path to saved model
# MODEL_PATH = os.getenv("MODEL_PATH", "/data/06_model_output/best_model.pk")
MODEL_PATH = os.getenv("MODEL_PATH", "/home/kedro_docker/model/best_model.pkl")

# Load model during server startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class PredictionRequest(BaseModel):
    """
    Defines the input data structure for making predictions.

    Attributes:
        data (dict): A dictionary containing the feature values for a Pokemon.
    """

    data: dict


@app.get("/")
def read_root():
    """
    Root endpoint to welcome users to the API.

    Returns:
        dict: A simple welcome message.
    """
    return {"message": "Welcome to the model API"}


@app.get("/check_model")
def check_model():
    """
    Endpoint to check if the model has been loaded successfully.

    Returns:
        dict: A success message if the model is loaded, else raises an HTTPException.
    """
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    print(f"Model type: {type(model)}")
    print(f"Model parameters: {model.get_params()}")

    return {"message": "Model loaded successfully"}


@app.post("/predict/")
def predict(request: PredictionRequest):
    """
    Endpoint for making Pokemon Legendary status predictions.

    Args:
        request (PredictionRequest): A JSON object containing Pokemon features.

    Returns:
        dict: A dictionary containing the prediction (0 for not Legendary, 1 for Legendary).

    Raises:
        HTTPException: If the model is not found or if there's an issue with the prediction.
    """
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # convert the input data to a DataFrame
    input_data = pd.DataFrame([request.data])

    # Making a prediction
    prediction = model.predict(input_data)

    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
