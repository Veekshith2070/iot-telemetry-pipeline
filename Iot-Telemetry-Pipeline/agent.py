import pandas as pd
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#============================================
# 1. KNOWLEDGE BASE & RAG SETUP (ChromaDB)
#============================================

print("---[Step 1: Initializing RAG Knowledge Base in ChromaDB] ---")

# Simulated Iot Hardware & Troubleshooting Manual

iot_repair_manual = """
DOC-101: Temperature Anamoly and Thermal Management
If a device reports a temperature exceeding 90C, this indicates cooling fan failure or heat sink
blockage
Immediate Action: Shut down primary motor , inspect cooling ducts, and check coolant pump pressure.

DOC-102: Exceesive Vibration and Mechanical Wear
If vibration levels exceed 3.0 mm/s, this indicates bearings and recalibrate rotor balancing weughts.
Immediate Action: Lubricate mechanical bearings and recalibrate rotor balancing weights.

DOC-103: Pressure Fluctuations
If operational pressure drops below 20 PSI, inspect intake valves for pneumatic seal leakage.
Immediate Action: Replace pneumatic valve 0-rings and verify compressor output.
"""

# Text Chunking: Breaking text into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.create_documents([iot_repair_manual])

#storing in CheomaDb Vecto Database
# (We use a 100-dim local embedding so it runs instantly without requiring external paid API keys)
embeddings = FakeEmbeddings(size=100)
vector_db = Chroma.from_documents(chunks, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k":1})
print(f"Indexed {len(chunks)} troubleshooting manual sections into ChromaDB.\n")

# =========================================================================
# 2. DEFINING THE AGENT'S TOOLS (Python Functions with @tool)
# =========================================================================
print("---[Step 2: Registering Agent Tools] ---")

@tool
def query_sensor_telemetry(device_id: str) -> str:
    """Useful to check the latest telemetry readings, temperature, vibration, and anomaly flags for a 
device."""
    try:
        df = pd.read_csv("processed_telemetry.csv")
    except FileNotFoundError:
        return "Telemetry database not found. please run pipeline.py first."

    device_data = df[df['device_id']==device_id]
    if device_data.empty:
        return f"Device {device_id} not found in telemetry records."

    # Grab the latest reading
    latest  = device_data.iloc[-1]
    has_anomaly = latest['is_anomaly']

    return (
        f"Telemetry for {device_id}: "
        f"Temperature = {latest['temperature']:.1f}°C, "
        f"Vibration = {latest['vibration']:.2f}, "
        f"Pressure = {latest['pressure']:.1f} PSI, "
        f"Anomaly Flagged = {has_anomaly}"
    )

@tool
def search_repair_manual(query: str) -> str:
    """Useful to search device repair mauals and troubleshooting instructions using semantic
    search."""
    docs = retriever.invoke(query)
    if docs:
        return docs[0].page_content
    return "No relevent repar procedure found in the manual for your query."
# The Agent's Toolbelt
tools = [query_sensor_telemetry, search_repair_manual]
print(f"Registered {len(tools)} tools: {[t.name for t in tools]}\n")

# =========================================================================
# 3. THE AGENTIC REASONING WORKFLOW (ReAct Simulation)
# =========================================================================
print("---[Step 3: Agentic Reasoning Workflow Setup] ---")

def run_diagnostic_agent(incident_alert: str, target_device: str):
    """
    Simulates the ReAct Agent Loop:
    Thought -> Action -> Observation -> Thought -> Action -> Observation -> Final Diagnosis
    """
    print(f"INCIDENT RECEIVED: '{incident_alert}'\n")

    # --- Step 1: Agent decides to check sensor data ---
    print("Agent Thought: An incident was reported. I must first inspect the live sensor readings.")
    print(f"Agent Action: calling tool query_sensor_telemetry('{target_device}')")
    telemetry_result = query_sensor_telemetry.invoke({"device_id": target_device})
    print(f"Agent Observation: {telemetry_result}")
    
    # --- Step 2: Agent reasons about what it observed ---
    print("Agent Thought: The telemetry shows an anomaly alert with high temperature/vibration. "
          "I need to search the ChromaDB repair manual for troubleshooting steps.")
    print("Agent Action: Calling tool search_repair_manual('Temperature Anomaly and Thermal Management')")
    repair_result = search_repair_manual.invoke({"query": "Temperature Anomaly and Thermal Management"})
    print(f"Agent Observation: {repair_result}\n")
    
    # --- Step 3: Agent synthesizes the Final Answer ---
    print("Agent Thought: I now have the live sensor evidence and the verified repair procedure. "
          "I am ready to generate the final diagnostic report.\n")
    
    final_report = f"""
=====================================================
          AUTOMATED IOT INCIDENT REPORT
=====================================================
Target Device:     {target_device}
Incident Summary:  {incident_alert}
Live Telemetry:    {telemetry_result}
Root Cause:        Thermal threshold violation detected by rolling Z-score.
Recommended Fix:   {repair_result.strip()}
Status:            CRITICAL - Maintenance Ticket Dispatched.
=====================================================
"""
    return final_report
if __name__ == "__main__":
    # Ensure processed_telemetry.csv exists by running pipeline.py if needed
    import os
    if not os.path.exists("processed_telemetry.csv"):
        import pipeline
        print("Generating fresh telemetry data first...")
        raw = pipeline.generate_sensor_telemetry()
        processed = pipeline.clean_and_process_telemetry(raw)
        processed.to_csv("processed_telemetry.csv", index=False)
    # Trigger the Agent to investigate an alert on DEV_101
    report = run_diagnostic_agent(
        incident_alert="Sensor alert triggered: Temperature deviation detected.",
        target_device="DEV_101"
    )
    print(report)