import os
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================================
# 1. KNOWLEDGE BASE & RAG SETUP (ChromaDB)
# =====================================================================
iot_repair_manual = """
DOC-101: Temperature Anomaly and Thermal Management
If a device reports temperature exceeding 80C, this indicates cooling fan failure or heat sink blockage.
Immediate Action: Shut down primary motor, clean air intake ducts, and verify coolant pump pressure.

DOC-102: Excessive Vibration and Mechanical Wear
If vibration levels exceed 2.2 mm/s, this indicates mechanical bearing misalignment or loose mounting.
Immediate Action: Tighten mounting bolts, apply mechanical bearing lubricant, and recalibrate balance.

DOC-103: Pressure Fluctuations and Pneumatic Seal
If operational pressure drops below 25 PSI, inspect pneumatic valves for seal leakage.
Immediate Action: Replace pneumatic valve O-rings and verify air compressor output.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.create_documents([iot_repair_manual])

# Vector Store
embeddings = FakeEmbeddings(size=100)
vector_db = Chroma.from_documents(chunks, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# =====================================================================
# 2. DEFINE AGENT TOOLS
# =====================================================================
@tool
def query_sensor_telemetry(device_id: str) -> str:
    """Useful to check the latest live sensor readings (temperature, vibration, pressure, anomaly status) for a given device_id."""
    try:
        df = pd.read_csv("processed_telemetry.csv")
    except FileNotFoundError:
        return "Telemetry database not found. Please run pipeline.py first."
    
    device_data = df[df['device_id'] == device_id]
    if device_data.empty:
        return f"Device {device_id} not found in database."
    
    latest = device_data.iloc[-1]
    return (
        f"Device: {device_id} | "
        f"Temperature: {latest['temperature']:.1f}°C | "
        f"Vibration: {latest['vibration']:.2f} mm/s | "
        f"Pressure: {latest['pressure']:.1f} PSI | "
        f"Anomaly Detected: {latest['is_anomaly']}"
    )

@tool
def search_repair_manual(symptom_or_error: str) -> str:
    """Useful to query the official device troubleshooting manual and error-code repair steps using semantic search."""
    docs = retriever.invoke(symptom_or_error)
    if docs:
        return docs[0].page_content
    return "No relevant repair procedure found in the manual."

tools = [query_sensor_telemetry, search_repair_manual]
tool_map = {t.name: t for t in tools}

# =====================================================================
# 3. INITIALIZE LIVE LLM WITH NATIVE TOOL CALLING (Groq Llama 3.3)
# =====================================================================
llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.0
)

# Bind the tools directly to Llama 3.3
llm_with_tools = llm.bind_tools(tools)

def run_agent_diagnosis(device_id: str, issue_description: str) -> str:
    """
    Autonomous ReAct Loop using Native Tool Calling:
    Sends user query -> LLM emits tool call -> Python executes tool -> LLM analyzes output -> Returns Final Diagnosis.
    """
    system_instruction = SystemMessage(
        content=(
            "You are an expert industrial IoT diagnostic agent. "
            "When an alert is received, first inspect the device's live sensor telemetry using your tools. "
            "If an anomaly is detected, search the repair manual for the recommended action. "
            "Finally, provide a clear diagnosis including: "
            "1. Live Sensor Readings, 2. Root Cause, and 3. Recommended Immediate Action."
        )
    )
    user_prompt = HumanMessage(
        content=f"Diagnose device '{device_id}'. Reported Alert: '{issue_description}'."
    )
    
    messages = [system_instruction, user_prompt]
    
    # Autonomous Tool-Calling Loop (up to 4 steps)
    for step in range(4):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # If the LLM didn't request any tools, it has reached its final answer!
        if not response.tool_calls:
            return response.content
        
        # The LLM requested one or more tool calls: execute them
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"\n[Agent Action] Calling Tool: '{tool_name}' with args {tool_args}")
            
            tool_fn = tool_map.get(tool_name)
            if tool_fn:
                tool_output = tool_fn.invoke(tool_args)
            else:
                tool_output = f"Tool {tool_name} not found."
                
            print(f"[Agent Observation] Output: {tool_output}")
            
            # Feed the observation back into the conversation
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
    return response.content

if __name__ == "__main__":
    print("\n--- Testing Live Llama-3.3 Agent with Native Tool Calling ---")
    diagnosis = run_agent_diagnosis("DEV_101", "High temperature warning flagged by monitoring system.")
    print("\n=====================================================")
    print("FINAL AGENT DIAGNOSIS:")
    print("=====================================================")
    print(diagnosis)