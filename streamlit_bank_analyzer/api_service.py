"""
Production-ready FastAPI microservice for payment analysis.
Provides REST endpoints for transaction analysis and recurring payment detection.
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from payment_analyzer import analyze_recurring_payments
from advanced_nlp import AdvancedNLPProcessor
from advanced_ml import AdvancedMLProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global processors
nlp_processor = None
ml_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global nlp_processor, ml_processor

    # Startup
    logger.info("Starting up the Payment Analysis API...")
    try:
        nlp_processor = AdvancedNLPProcessor()
        ml_processor = AdvancedMLProcessor()
        logger.info("Successfully initialized NLP and ML processors")
    except Exception as e:
        logger.error(f"Failed to initialize processors: {e}")
        # Continue without advanced processors

    yield

    # Shutdown
    logger.info("Shutting down the Payment Analysis API...")

# Create FastAPI app
app = FastAPI(
    title="Payment Analysis API",
    description="Advanced ML-powered payment analysis and recurring payment detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TransactionRequest(BaseModel):
    """Request model for transaction analysis."""
    description: str = Field(..., description="Transaction description")
    amount_str: str = Field(..., description="Transaction amount as string")
    formatted_date: Optional[str] = Field(None, description="Transaction date")

    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError('Description cannot be empty')
        return v.strip()

    @validator('amount_str')
    def validate_amount(cls, v):
        if not v or not v.strip():
            raise ValueError('Amount cannot be empty')
        return v.strip()

class BatchTransactionRequest(BaseModel):
    """Request model for batch transaction analysis."""
    transactions: List[TransactionRequest] = Field(..., description="List of transactions to analyze")
    user_name: Optional[str] = Field(None, description="User identifier")

    @validator('transactions')
    def validate_transactions(cls, v):
        if not v:
            raise ValueError('At least one transaction is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 transactions allowed per request')
        return v

class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    recurring_payments: List[Dict[str, Any]] = Field(..., description="Detected recurring payments")
    recommendations: List[str] = Field(..., description="IBAN change recommendations")
    total_analyzed: int = Field(..., description="Total transactions analyzed")
    unique_descriptions: int = Field(..., description="Number of unique transaction descriptions")
    processing_time: float = Field(..., description="Processing time in seconds")
    ml_enhanced: bool = Field(..., description="Whether ML enhancement was used")

class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction."""
    entities: Dict[str, List[str]] = Field(..., description="Extracted entities")
    transaction_type: Dict[str, float] = Field(..., description="Predicted transaction types")
    confidence: float = Field(..., description="Overall confidence score")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    processors: Dict[str, bool] = Field(..., description="Processor availability")

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    processors_status = {
        "nlp_processor": nlp_processor is not None,
        "ml_processor": ml_processor is not None
    }

    return HealthResponse(
        status="healthy" if all(processors_status.values()) else "degraded",
        version="1.0.0",
        processors=processors_status
    )

@app.post("/analyze/transactions", response_model=AnalysisResponse)
async def analyze_transactions(request: BatchTransactionRequest, background_tasks: BackgroundTasks):
    """
    Analyze transactions for recurring payments.

    This endpoint uses advanced ML techniques to identify recurring payment patterns
    and provides recommendations for IBAN changes.
    """
    start_time = datetime.now()

    try:
        # Convert request to internal format
        transactions = []
        for tx in request.transactions:
            transaction_dict = {
                'description': tx.description,
                'amount_str': tx.amount_str,
                'formatted_date': tx.formatted_date or datetime.now().strftime('%d.%m.%Y')
            }
            transactions.append(transaction_dict)

        # Perform analysis
        results = await asyncio.get_event_loop().run_in_executor(
            None, analyze_recurring_payments, transactions
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Add ML enhancement flag
        results['processing_time'] = processing_time
        results['ml_enhanced'] = nlp_processor is not None and ml_processor is not None

        return AnalysisResponse(**results)

    except Exception as e:
        logger.error(f"Error analyzing transactions: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/extract/entities", response_model=EntityExtractionResponse)
async def extract_entities(description: str = Query(..., description="Transaction description to analyze")):
    """
    Extract entities from a transaction description.

    Uses advanced NLP techniques to identify merchants, contract IDs, amounts, and payment types.
    """
    if not nlp_processor:
        raise HTTPException(status_code=503, detail="NLP processor not available")

    try:
        # Extract entities
        entities = nlp_processor.extract_entities(description)

        # Classify transaction type
        transaction_types = nlp_processor.classify_transaction_type(description)

        # Calculate confidence
        confidence = max(transaction_types.values()) if transaction_types else 0.0

        return EntityExtractionResponse(
            entities=entities,
            transaction_type=transaction_types,
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")

@app.post("/analyze/similarity")
async def calculate_similarity(
    text1: str = Query(..., description="First transaction description"),
    text2: str = Query(..., description="Second transaction description")
):
    """
    Calculate semantic similarity between two transaction descriptions.

    Returns a similarity score between 0 and 1, where 1 means identical and 0 means completely different.
    """
    if not nlp_processor:
        raise HTTPException(status_code=503, detail="NLP processor not available")

    try:
        similarity = nlp_processor.calculate_semantic_similarity(text1, text2)

        return {
            "similarity_score": similarity,
            "interpretation": "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
        }

    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get API usage statistics."""
    # In a real implementation, you'd track metrics with Prometheus/statsd
    return {
        "uptime": "Service running",
        "total_requests": 0,  # Would be tracked in production
        "average_response_time": 0.0,
        "error_rate": 0.0
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "type": "http_exception"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": "internal_error"}
    )

# Startup function for running the server
def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the FastAPI server.

    Args:
        host: Host address to bind to
        port: Port to listen on
    """
    logger.info(f"Starting Payment Analysis API on {host}:{port}")
    uvicorn.run(
        "api_service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start_api_server()
