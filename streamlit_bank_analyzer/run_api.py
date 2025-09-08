#!/usr/bin/env python3
"""
Script to run the FastAPI server for Bank Statement Analyzer.
"""

import uvicorn
from api import app

if __name__ == "__main__":
    print("Starting Bank Statement Analyzer API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive API docs at: http://localhost:8000/docs")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
