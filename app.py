from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI()

# Path to saved model
MODEL_PATH = os.getenv("MODEL_PATH", "/data/06_model_output/best_model.pk")

# Load model during server startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class PredictionRequest(BaseModel):
    data: dict


@app.get("/")
def read_root():
    return {"message": "Welcome to the model API"}


@app.post("/predict/")
def predict(request: PredictionRequest):
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
