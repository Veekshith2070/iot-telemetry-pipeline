from pydantic import BaseModel, Field
from typing import Optional, List

class DiagnosticRequest(BaseModel):
    device_id: str = Field(..., example="DEV_101", description="Unique identifier for the IoT device.")
    issue_description: Optional[str] = Field(
        default="Route automated diagnostic check",
        example="Sensor alert: High temperature deviation detected."
    )

class IncidentReport(BaseModel):
    device_id: str
    status: str = Field(..., description="NORMAL. WARNING or CRITICAL")
    temperature: float
    vibration: float
    pressure: float
    anomaly_detected: bool
    root_cause: str
    recommended_action: str
    maintenance_ticket_id: str


