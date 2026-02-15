"""
FastAPI application for serving ML model predictions.

Run with: uvicorn api:app --reload
Access docs at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Request 
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import logging
import pandas as pd
from datetime import datetime
from src.train import load_model
from src.predict import predict, _set_model_cache
from config import FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="API for predicting employee insurance enrollment using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Available models
AVAILABLE_MODELS = ['logistic_regression', 'random_forest', 'lightgbm']
DEFAULT_MODEL = 'logistic_regression'

# Cache for loaded models
loaded_models = {}


@app.on_event("startup")
async def startup_event():
    """Initialize model cache on startup."""
    logger.info("=" * 80)
    logger.info("STARTUP: Loading models...")
    logger.info("=" * 80)
    
    # Pre-load all models into the predict module cache
    for model_name in AVAILABLE_MODELS:
        try:
            loaded_models[model_name] = load_model(model_name)
            # 🔍 LOG: Show model ID at startup
            logger.info(f"✓ Pre-loaded model: {model_name} (id={id(loaded_models[model_name])})")
        except Exception as e:
            logger.warning(f"Could not pre-load {model_name}: {e}")
    
    # Sync cache to predict module
    _set_model_cache(loaded_models)
    logger.info(f"✓ Model cache synchronized: {list(loaded_models.keys())}")
    logger.info("=" * 80)

def get_model(model_name: str):
    """Load and cache model on demand."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {AVAILABLE_MODELS}")
    
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = load_model(model_name)
            logger.info(f"Model '{model_name}' loaded successfully")
            
            # Sync to predict module cache
            _set_model_cache(loaded_models)
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' not available")
    
    return loaded_models[model_name]


# ===== REQUEST/RESPONSE MODELS =====

class EmployeeInput(BaseModel):
    """Input schema for single employee prediction."""
    has_dependents: str = Field(..., example="Yes", description="Whether employee has dependents (Yes/No)")
    employment_type: str = Field(..., example="Full-time", description="Employment type")
    age: int = Field(..., ge=18, le=100, example=35, description="Employee age")
    salary: float = Field(..., gt=0, example=75000.0, description="Annual salary")
    model: Optional[str] = Field(DEFAULT_MODEL, description="Model to use for prediction")
    
    @validator('has_dependents')
    def validate_dependents(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('has_dependents must be "Yes" or "No"')
        return v
    
    @validator('employment_type')
    def validate_employment(cls, v):
        valid_types = ['Full-time', 'Part-time', 'Contract']
        if v not in valid_types:
            raise ValueError(f'employment_type must be one of {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "has_dependents": "Yes",
                "employment_type": "Full-time",
                "age": 35,
                "salary": 75000.0
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    prediction: int = Field(..., description="Predicted class (0=No, 1=Yes)")
    probability: float = Field(..., description="Probability of enrollment")
    enrolled: str = Field(..., description="Human-readable prediction")
    confidence: str = Field(..., description="Confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    employees: List[EmployeeInput] = Field(..., min_items=1, max_items=1000)
    model: Optional[str] = Field(DEFAULT_MODEL, description="Model to use for predictions")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    summary: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]
    available_models: List[str]
    default_model: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    available_models: List[str]
    default_model: str
    features: List[str]
    categorical_features: List[str]
    numeric_features: List[str]
    model_specific_notes: dict


# ===== HELPER FUNCTIONS =====

def get_confidence_level(probability: float) -> str:
    """Convert probability to confidence level."""
    if probability >= 0.8:
        return "Very High"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    elif probability >= 0.2:
        return "Low"
    else:
        return "Very Low"


def make_single_prediction(employee: EmployeeInput) -> PredictionResponse:
    """Make prediction for single employee."""
    model_name = employee.model or DEFAULT_MODEL
    
    # 🔍 LOG 1: Show which model was requested
    logger.info(f"=== PREDICTION: Requested model='{model_name}' ===")
    
    # Get model (loads on demand if not cached)
    model = get_model(model_name)
    
    # 🔍 LOG 2: Show which model object we got (by ID)
    logger.info(f"=== PREDICTION: Using model object id={id(model)} ===")
    
    # Convert to DataFrame
    data = pd.DataFrame([{
        'has_dependents': employee.has_dependents,
        'employment_type': employee.employment_type,
        'age': employee.age,
        'salary': employee.salary
    }])
    
    # Make prediction
    predictions, probabilities = predict(model_name, data)
    
    pred = int(predictions[0])
    prob = float(probabilities[0, 1])
    
    # 🔍 LOG 3: Show the result
    logger.info(f"=== PREDICTION: Result pred={pred}, prob={prob:.4f} ===")
    
    return PredictionResponse(
        prediction=pred,
        probability=round(prob, 4),
        enrolled="Yes" if pred == 1 else "No",
        confidence=get_confidence_level(prob),
        timestamp=datetime.now().isoformat()
    )


# ===== API ENDPOINTS =====

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and model availability.
    """
    # Check which models are loaded
    loaded = list(loaded_models.keys())
    
    return HealthResponse(
        status="healthy" if len(loaded) > 0 else "degraded",
        models_loaded=loaded,
        available_models=AVAILABLE_MODELS,
        default_model=DEFAULT_MODEL,
        timestamp=datetime.now().isoformat()
    )


@app.get("/models", response_model=List[str], tags=["System"])
async def list_models():
    """
    List available models.
    
    Returns list of model names that can be used for predictions.
    """
    return ["logistic_regression", "random_forest", "lightgbm"]


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """
    Get model information.
    
    Returns details about available models, features, and preprocessing configuration.
    """
    return ModelInfoResponse(
        available_models=["logistic_regression", "random_forest", "lightgbm"],
        default_model="logistic_regression",
        features=FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        numeric_features=NUMERIC_FEATURES,
        model_specific_notes={
            "logistic_regression": "Uses OneHotEncoder + StandardScaler. Best for linear relationships and interpretability.",
            "random_forest": "Uses OrdinalEncoder + passthrough (no scaling). Best for non-linear patterns and feature interactions.",
            "lightgbm": "Uses OrdinalEncoder + passthrough (no scaling). Best for complex patterns with L1/L2 regularization."
        }
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(employee: EmployeeInput):
    """
    Make prediction for a single employee.
    
    Returns enrollment prediction with probability and confidence level.
    
    Example request:
    ```json
    {
        "has_dependents": "Yes",
        "employment_type": "Full-time",
        "age": 35,
        "salary": 75000
    }
    ```
    """
    try:
        return make_single_prediction(employee)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple employees.
    
    Accepts up to 1000 employees per request.
    Returns predictions with summary statistics.
    
    Example request:
    ```json
    {
        "employees": [
            {
                "has_dependents": "Yes",
                "employment_type": "Full-time",
                "age": 35,
                "salary": 75000
            },
            {
                "has_dependents": "No",
                "employment_type": "Part-time",
                "age": 25,
                "salary": 35000
            }
        ],
        "model": "logistic_regression"
    }
    ```
    """
    try:
        # Use specified model or default
        model_name = request.model or DEFAULT_MODEL
        model = get_model(model_name)
        
        predictions = []
        for employee in request.employees:
            # Override model for each employee
            employee.model = model_name
            pred = make_single_prediction(employee)
            predictions.append(pred)
        
        # Calculate summary
        total = len(predictions)
        enrolled = sum(1 for p in predictions if p.prediction == 1)
        not_enrolled = total - enrolled
        avg_prob = sum(p.probability for p in predictions) / total
        
        summary = {
            "total_employees": total,
            "predicted_enrolled": enrolled,
            "predicted_not_enrolled": not_enrolled,
            "average_probability": round(avg_prob, 4),
            "enrollment_rate": round(enrolled / total * 100, 2)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain", response_model=dict, tags=["Predictions"])
async def predict_with_explanation(employee: EmployeeInput):
    """
    Make prediction with feature importance explanation.
    
    Returns prediction along with feature contributions (if model supports it).
    """
    try:
        # Make prediction
        prediction = make_single_prediction(employee)
        
        # Get model for feature importance
        model_name = employee.model or DEFAULT_MODEL
        model = get_model(model_name)
        
        # Get feature importance from model
        if hasattr(model.named_steps['classifier'], 'coef_'):
            # For logistic regression, get coefficients
            preprocessor = model.named_steps['preprocessor']
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
            all_features = list(cat_features) + NUMERIC_FEATURES
            coefficients = model.named_steps['classifier'].coef_[0]
            
            # Map to input features
            feature_importance = dict(zip(all_features, coefficients.tolist()))
            
            return {
                "prediction": prediction.dict(),
                "feature_importance": feature_importance,
                "note": "Positive values increase enrollment probability, negative values decrease it"
            }
        else:
            return {
                "prediction": prediction.dict(),
                "note": "Feature importance not available for this model type"
            }
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== ERROR HANDLERS =====

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)