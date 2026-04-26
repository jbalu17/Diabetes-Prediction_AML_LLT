"""FastAPI application for diabetes prediction."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import time

from api.schemas import (
    DiabetesInput, PredictionResponse, BatchInput, BatchResponse,
    HealthResponse, ModelInfoResponse
)
from src.predict import get_predictor, DiabetesPredictor
from src.utils import get_feature_names

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="""
    A machine learning API for predicting diabetes risk based on patient health metrics.
    
    ## Features
    - Single and batch predictions
    - Probability scores and confidence levels
    - Interactive documentation
    
    ## Model
    - Algorithm: Random Forest Classifier
    - Dataset: Pima Indians Diabetes Database
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))
templates.env.cache = {}   # 🔥 ADD THIS LINE

# Metrics tracking
metrics = {
    "total_predictions": 0,
    "start_time": time.time()
}


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        get_predictor()
        print("✓ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the frontend interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health status of the API."""
    try:
        predictor = get_predictor()
        return HealthResponse(status="healthy", model_loaded=True)
    except Exception:
        return HealthResponse(status="unhealthy", model_loaded=False)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Get information about the loaded model."""
    return ModelInfoResponse(
        model_type="RandomForestClassifier",
        features=get_feature_names(),
        version="1.0.0"
    )


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Get API usage metrics."""
    uptime = time.time() - metrics["start_time"]
    return {
        "total_predictions": metrics["total_predictions"],
        "uptime_seconds": round(uptime, 2),
        "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: DiabetesInput):
    """
    Make a single diabetes prediction.
    
    Takes patient health metrics and returns the prediction with probability scores.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(input_data.model_dump())
        metrics["total_predictions"] += 1
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch_input: BatchInput):
    """
    Make batch predictions for multiple patients.
    
    Takes a list of patient health metrics and returns predictions for all.
    """
    try:
        predictor = get_predictor()
        features_list = [item.model_dump() for item in batch_input.inputs]
        results = predictor.predict_batch(features_list)
        metrics["total_predictions"] += len(results)
        return BatchResponse(
            predictions=[PredictionResponse(**r) for r in results],
            count=len(results)
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
