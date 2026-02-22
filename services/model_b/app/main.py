from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import logging
import time

from app.model import ModelService, MODEL_VERSION
from app.schemas import PredictionRequest, PredictionResponse
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model B Service")

model_service = ModelService()


@app.on_event("startup")
def startup_event():
    logger.info("Loading model...")
    model_service.load_model()
    logger.info("Model loaded successfully.")


@app.get("/health") #Is service alive?
def health():
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "version": MODEL_VERSION
    }


@app.post("/predict", response_model=PredictionResponse) #Run model
def predict(request: PredictionRequest):
    REQUEST_COUNT.inc()

    try:
       

        prediction, inference_time = model_service.predict(request.features)
        REQUEST_LATENCY.observe(inference_time / 1000)

        return PredictionResponse(
            prediction=prediction,
            model_version=MODEL_VERSION,
            inference_time_ms=inference_time
        )

    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/metrics") #Monitoring endpoint
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)