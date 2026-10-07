import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agent import run_agent_diagnosis

app = FastAPI(
    title="Autonomous IoT Diagnostic & Telemetry Microservice",
    description="An AI-powered agentic microservice that monitors IoT sensor telemetry and diagnoses machine faults using RAG.",
    version="1.0.0"
)

# Request Schema
class DiagnosticRequest(BaseModel):
    device_id: str = Field(..., example="DEV_101", description="Target IoT Device ID")
    issue_description: str = Field(
        default="Automated routine diagnostic scan",
        example="High temperature alert detected by monitoring system."
    )

# Endpoints
@app.get("/")
def root():
    return {
        "service": "Autonomous IoT Diagnostic Agent",
        "status": "ONLINE",
        "endpoints": {
            "docs": "/docs",
            "anomalies": "/telemetry/anomalies",
            "diagnose": "/diagnose"
        }
    }

@app.get("/telemetry/anomalies")
def get_detected_anomalies():
    """Returns all sensor telemetry records that have been flagged as statistical anomalies by pipeline.py."""
    if not os.path.exists("processed_telemetry.csv"):
        raise HTTPException(status_code=404, detail="Telemetry data not found. Please run pipeline.py first.")
    
    df = pd.read_csv("processed_telemetry.csv")
    anomalies = df[df['is_anomaly'] == True]
    
    return {
        "total_anomalies_detected": len(anomalies),
        "anomalous_records": anomalies.tail(10).to_dict(orient="records")
    }

@app.post("/diagnose")
def trigger_agent_diagnosis(request: DiagnosticRequest):
    """Triggers the autonomous LangChain ReAct agent to inspect device telemetry and generate an incident report."""
    try:
        diagnosis_report = run_agent_diagnosis(
            device_id=request.device_id,
            issue_description=request.issue_description
        )
        return {
            "device_id": request.device_id,
            "status": "SUCCESS",
            "incident_report": diagnosis_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)