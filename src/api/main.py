"""
Advanced FastAPI Web Application for Real-time Customer Analytics
Provides sophisticated API endpoints for customer analytics and optimization.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime, timedelta
import redis
import pickle
from contextlib import asynccontextmanager

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.data_simulator import AdvancedCustomerDataSimulator
from causal_inference.causal_analysis import AdvancedCausalAnalyzer
from predictive_modeling.model_training import AdvancedPredictiveModeler
from optimization.optimization_engine import AdvancedOptimizationEngine


# Pydantic models for API requests/responses
class CustomerData(BaseModel):
    customer_id: int
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    education_level: str
    region: str
    marital_status: str
    monthly_spend: float = Field(..., ge=0)
    loyalty_score: float = Field(..., ge=0, le=1)
    online_activity: int = Field(..., ge=0)
    customer_segment: str


class TargetingRequest(BaseModel):
    customer_features: Dict[str, Any]
    optimization_method: str = "ensemble"
    budget_constraint: Optional[float] = None


class AnalyticsRequest(BaseModel):
    analysis_type: str  # "causal", "predictive", "optimization"
    parameters: Dict[str, Any] = {}


class PredictionRequest(BaseModel):
    customer_data: CustomerData
    model_type: str = "ensemble"


# Global variables for caching and state management
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
cache_ttl = 3600  # 1 hour

# Global instances
customer_simulator = None
causal_analyzer = None
predictive_modeler = None
optimization_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global customer_simulator, causal_analyzer, predictive_modeler, optimization_engine
    
    # Initialize components
    print("🚀 Initializing RazorVine Analytics Platform...")
    
    # Generate sample data if not exists
    if not os.path.exists("data/simulated_customer_data.csv"):
        print("📊 Generating sample customer data...")
        customer_simulator = AdvancedCustomerDataSimulator(
            n_customers=10000,
            n_days=60,
            seed=42
        )
        customers_df, interactions_df, summary_stats = customer_simulator.run_full_simulation()
        customers_df.to_csv("data/simulated_customer_data.csv", index=False)
        interactions_df.to_csv("data/simulated_interaction_data.csv", index=False)
    
    # Load data
    data = pd.read_csv("data/simulated_customer_data.csv")
    
    # Initialize analyzers (lazy loading)
    print("✅ Platform initialized successfully!")
    
    yield
    
    # Cleanup
    print("🔄 Cleaning up resources...")


# Create FastAPI app
app = FastAPI(
    title="RazorVine Analytics Platform",
    description="Advanced Customer Analytics and Optimization Platform",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Utility functions
def get_cached_data(key: str) -> Optional[Any]:
    """Get data from Redis cache."""
    try:
        data = redis_client.get(key)
        return pickle.loads(data) if data else None
    except:
        return None


def set_cached_data(key: str, data: Any, ttl: int = cache_ttl):
    """Set data in Redis cache."""
    try:
        redis_client.setex(key, ttl, pickle.dumps(data))
    except:
        pass


def load_data() -> pd.DataFrame:
    """Load customer data."""
    return pd.read_csv("data/simulated_customer_data.csv")


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RazorVine Analytics Platform</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .endpoint { background: rgba(255,255,255,0.1); padding: 20px; margin: 20px 0; border-radius: 10px; }
            .method { display: inline-block; padding: 5px 10px; border-radius: 5px; margin-right: 10px; }
            .get { background: #61affe; }
            .post { background: #49cc90; }
            .put { background: #fca130; }
            .delete { background: #f93e3e; }
            code { background: rgba(0,0,0,0.3); padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 RazorVine Analytics Platform</h1>
                <p>Advanced Customer Analytics and Optimization API</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/api/health</code>
                <p>Check API health and status</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/api/data/summary</code>
                <p>Get customer data summary statistics</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <code>/api/analytics/causal</code>
                <p>Run causal inference analysis</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <code>/api/analytics/predictive</code>
                <p>Run predictive modeling analysis</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <code>/api/optimization/targeting</code>
                <p>Get customer targeting recommendations</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/docs</code>
                <p>Interactive API documentation</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "redis": "connected" if redis_client.ping() else "disconnected",
            "data": "available" if os.path.exists("data/simulated_customer_data.csv") else "unavailable"
        }
    }


@app.get("/api/data/summary")
async def get_data_summary():
    """Get customer data summary statistics."""
    cache_key = "data_summary"
    cached_data = get_cached_data(cache_key)
    
    if cached_data:
        return cached_data
    
    try:
        data = load_data()
        
        summary = {
            "total_customers": len(data),
            "customer_segments": data['customer_segment'].value_counts().to_dict(),
            "average_clv": float(data['customer_lifetime_value'].mean()),
            "average_churn_risk": float(data['churn_risk'].mean()),
            "promotion_response_rate": float(data['promotion_response'].mean()),
            "average_monthly_spend": float(data['monthly_spend'].mean()),
            "data_quality": {
                "missing_values": data.isnull().sum().to_dict(),
                "duplicates": int(data.duplicated().sum())
            },
            "timestamp": datetime.now().isoformat()
        }
        
        set_cached_data(cache_key, summary)
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.post("/api/analytics/causal")
async def run_causal_analysis(request: AnalyticsRequest, background_tasks: BackgroundTasks):
    """Run causal inference analysis."""
    cache_key = f"causal_analysis_{hash(str(request.parameters))}"
    cached_result = get_cached_data(cache_key)
    
    if cached_result:
        return cached_result
    
    try:
        data = load_data()
        
        # Initialize analyzer
        analyzer = AdvancedCausalAnalyzer(
            data=data,
            treatment_col='promotion_response',
            outcome_col='customer_lifetime_value'
        )
        
        # Run analyses based on parameters
        results = {}
        
        if request.parameters.get('propensity_matching', True):
            results['propensity_matching'] = analyzer.propensity_score_matching()
        
        if request.parameters.get('ml_based', True):
            results['ml_based'] = analyzer.ml_based_causal_inference()
        
        if request.parameters.get('sensitivity_analysis', True):
            results['sensitivity_analysis'] = analyzer.sensitivity_analysis()
        
        # Generate report
        report = analyzer.generate_report()
        
        result = {
            "analysis_type": "causal_inference",
            "results": results,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        set_cached_data(cache_key, result)
        
        # Background task to save visualizations
        background_tasks.add_task(save_causal_visualizations, analyzer)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in causal analysis: {str(e)}")


@app.post("/api/analytics/predictive")
async def run_predictive_analysis(request: AnalyticsRequest):
    """Run predictive modeling analysis."""
    cache_key = f"predictive_analysis_{hash(str(request.parameters))}"
    cached_result = get_cached_data(cache_key)
    
    if cached_result:
        return cached_result
    
    try:
        data = load_data()
        
        # Initialize modeler
        modeler = AdvancedPredictiveModeler(
            data=data,
            target_col=request.parameters.get('target_col', 'customer_lifetime_value'),
            problem_type=request.parameters.get('problem_type', 'regression')
        )
        
        # Run analyses
        results = {}
        
        if request.parameters.get('ensemble_models', True):
            results['ensemble_models'] = modeler.train_ensemble_models()
        
        if request.parameters.get('deep_learning', True):
            results['deep_learning'] = modeler.train_deep_learning_model()
        
        if request.parameters.get('feature_selection', True):
            results['feature_selection'] = modeler.feature_selection()
        
        if request.parameters.get('model_interpretation', True):
            results['model_interpretation'] = modeler.model_interpretation()
        
        # Generate report
        report = modeler.generate_report()
        
        result = {
            "analysis_type": "predictive_modeling",
            "results": results,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        set_cached_data(cache_key, result)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in predictive analysis: {str(e)}")


@app.post("/api/optimization/targeting")
async def get_targeting_recommendations(request: TargetingRequest):
    """Get customer targeting recommendations."""
    try:
        data = load_data()
        
        # Initialize optimization engine
        engine = AdvancedOptimizationEngine(customer_data=data)
        
        # Run optimization if not cached
        cache_key = "optimization_results"
        cached_results = get_cached_data(cache_key)
        
        if not cached_results:
            engine.run_comprehensive_optimization()
            set_cached_data(cache_key, engine.results)
        else:
            engine.results = cached_results
        
        # Get targeting strategy
        strategy = engine.customer_targeting_strategy(request.customer_features)
        
        # Add optimization context
        result = {
            "customer_features": request.customer_features,
            "targeting_strategy": strategy,
            "optimization_method": request.optimization_method,
            "budget_constraint": request.budget_constraint,
            "recommendations": {
                "promotion_type": _get_promotion_recommendation(strategy),
                "expected_roi": _calculate_roi(strategy, request.budget_constraint),
                "confidence_score": strategy.get('bandit_confidence', 0.5)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in targeting optimization: {str(e)}")


@app.post("/api/predictions/customer")
async def predict_customer_outcomes(request: PredictionRequest):
    """Predict customer outcomes using trained models."""
    try:
        data = load_data()
        
        # Initialize modeler
        modeler = AdvancedPredictiveModeler(
            data=data,
            target_col='customer_lifetime_value',
            problem_type='regression'
        )
        
        # Train models if not cached
        cache_key = "trained_models"
        cached_models = get_cached_data(cache_key)
        
        if not cached_models:
            modeler.train_ensemble_models()
            modeler.train_deep_learning_model()
            set_cached_data(cache_key, modeler.models)
        else:
            modeler.models = cached_models
        
        # Prepare customer data
        customer_df = pd.DataFrame([request.customer_data.dict()])
        customer_processed = modeler.preprocessor.transform(customer_df)
        
        # Make predictions
        predictions = {}
        for model_name, model_result in modeler.models.items():
            if 'model' in model_result:
                pred = model_result['model'].predict(customer_processed)[0]
                predictions[model_name] = float(pred)
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        
        result = {
            "customer_id": request.customer_data.customer_id,
            "predictions": predictions,
            "ensemble_prediction": float(ensemble_pred),
            "model_type": request.model_type,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")


@app.get("/api/analytics/real-time")
async def get_real_time_analytics():
    """Get real-time analytics dashboard data."""
    try:
        data = load_data()
        
        # Calculate real-time metrics
        current_time = datetime.now()
        
        # Simulate real-time data updates
        recent_data = data.sample(frac=0.1)  # Simulate recent transactions
        
        real_time_metrics = {
            "current_time": current_time.isoformat(),
            "active_customers": len(recent_data),
            "revenue_today": float(recent_data['monthly_spend'].sum() / 30),  # Daily revenue
            "conversion_rate": float(recent_data['promotion_response'].mean()),
            "average_session_duration": np.random.randint(300, 1800),  # Simulated
            "top_performing_segments": recent_data['customer_segment'].value_counts().head(3).to_dict(),
            "churn_alerts": len(recent_data[recent_data['churn_risk'] > 0.7]),
            "trending_products": ["Product A", "Product B", "Product C"]  # Simulated
        }
        
        return real_time_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in real-time analytics: {str(e)}")


# Utility functions for targeting
def _get_promotion_recommendation(strategy: Dict) -> str:
    """Get promotion recommendation based on strategy."""
    bandit_action = strategy.get('bandit_action', 0)
    rl_action = strategy.get('rl_action', 0)
    
    # Combine recommendations
    combined_action = (bandit_action + rl_action) // 2
    
    promotion_types = [
        "No Promotion",
        "Discount (10%)",
        "Cashback (5%)",
        "Free Shipping",
        "Premium Service",
        "Exclusive Access"
    ]
    
    return promotion_types[min(combined_action, len(promotion_types) - 1)]


def _calculate_roi(strategy: Dict, budget_constraint: Optional[float]) -> float:
    """Calculate expected ROI."""
    expected_value = strategy.get('expected_value', 1000)
    base_value = 1000  # Assuming base CLV
    
    roi = (expected_value - base_value) / base_value * 100
    
    if budget_constraint:
        # Adjust ROI based on budget constraints
        roi = min(roi, budget_constraint * 0.1)  # Cap ROI at 10% of budget
    
    return float(roi)


async def save_causal_visualizations(analyzer: AdvancedCausalAnalyzer):
    """Save causal analysis visualizations."""
    try:
        analyzer.plot_results("static/causal_analysis_results.html")
    except Exception as e:
        print(f"Error saving visualizations: {e}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 