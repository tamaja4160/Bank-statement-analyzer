"""
FastAPI REST Service for Bank Statement Analyzer
Provides REST endpoints for generating statements, OCR processing, and payment analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from pathlib import Path
import json

# Import existing modules
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from statement_generator import generate_statements_for_name
from payment_analyzer import analyze_recurring_payments
from simple_image_reader.simple_image_reader import SimpleImageReader

app = FastAPI(
    title="Bank Statement Analyzer API",
    description="REST API for generating, processing, and analyzing bank statements",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session store for API (not using Streamlit session)
api_sessions = {}

# Pydantic models for request/response
class GenerateStatementsRequest(BaseModel):
    user_name: str
    num_statements: int = 5

class ProcessOCRRequest(BaseModel):
    user_name: str

class AnalyzePaymentsRequest(BaseModel):
    user_name: str

class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    balance: Optional[float] = None

class OCRResult(BaseModel):
    success: bool
    transactions: List[Transaction] = []
    error: Optional[str] = None

class AnalysisResult(BaseModel):
    recurring_payments: List[Dict] = []
    recommendations: List[str] = []

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bank Statement Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "generate_statements": "/generate-statements",
            "process_ocr": "/process-ocr",
            "analyze_payments": "/analyze-payments"
        }
    }

@app.post("/generate-statements", response_model=Dict[str, str])
async def generate_statements(request: GenerateStatementsRequest):
    """Generate bank statements for a user."""
    try:
        if not request.user_name.strip():
            raise HTTPException(status_code=400, detail="User name cannot be empty")

        # Generate statements
        image_paths = generate_statements_for_name(request.user_name, request.num_statements)

        if not image_paths:
            raise HTTPException(status_code=500, detail="Failed to generate statements")

        # Save to API session
        api_sessions[request.user_name] = {
            'image_paths': image_paths,
            'results': [],
            'analysis_results': None
        }

        return {
            "message": f"Successfully generated {len(image_paths)} statements",
            "user_name": request.user_name,
            "num_statements": len(image_paths)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating statements: {str(e)}")

@app.post("/process-ocr", response_model=Dict[str, List[OCRResult]])
async def process_ocr(request: ProcessOCRRequest):
    """Process existing images with OCR."""
    try:
        # Get image paths from API session
        user_session = api_sessions.get(request.user_name)
        if not user_session or not user_session.get('image_paths'):
            raise HTTPException(status_code=404, detail="No images found for user. Generate statements first.")

        image_paths = user_session['image_paths']

        # Process with OCR
        reader = SimpleImageReader()
        all_results = []

        for image_path in image_paths:
            result = reader.process_single_image(Path(image_path))
            ocr_result = OCRResult(
                success=result.get('success', False),
                transactions=[
                    Transaction(
                        date=tx.get('date', ''),
                        description=tx.get('description', ''),
                        amount=tx.get('amount', 0.0),
                        balance=tx.get('balance')
                    ) for tx in result.get('transactions', [])
                ],
                error=result.get('error')
            )
            all_results.append(ocr_result)

        # Save results to API session
        user_session['results'] = [result.dict() for result in all_results]
        user_session['analysis_results'] = None  # Reset analysis

        return {
            "results": all_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {str(e)}")

@app.post("/analyze-payments", response_model=AnalysisResult)
async def analyze_payments(request: AnalyzePaymentsRequest):
    """Analyze recurring payments for a user."""
    try:
        # Get results from API session
        user_session = api_sessions.get(request.user_name)
        if not user_session or not user_session.get('results'):
            raise HTTPException(status_code=404, detail="No OCR results found for user. Process OCR first.")

        # Collect all transactions
        all_transactions = []
        for result in user_session['results']:
            if result.get('success', False):
                all_transactions.extend(result.get('transactions', []))

        if not all_transactions:
            raise HTTPException(status_code=404, detail="No transactions found to analyze")

        # Analyze recurring payments
        analysis_results = analyze_recurring_payments(all_transactions)

        # Save analysis results to session
        user_session['analysis_results'] = analysis_results

        return AnalysisResult(
            recurring_payments=analysis_results.get('recurring_payments', []),
            recommendations=analysis_results.get('recommendations', [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing payments: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
