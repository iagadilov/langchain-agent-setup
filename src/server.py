"""
Fermer Agent API Server
=======================
FastAPI server exposing the LangGraph workflow via HTTP.

Replaces n8n webhook trigger with FastAPI endpoint.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import logging

from graph import process_message, fermer_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== API MODELS ==============

class WazzupMessage(BaseModel):
    """Wazzup webhook message format."""
    chatId: str
    channelId: str
    text: Optional[str] = None
    status: str = "inbound"
    chatType: str = "whatsapp"
    contentUri: Optional[str] = None  # For media messages


class WazzupWebhook(BaseModel):
    """Wazzup webhook payload format."""
    messages: list[WazzupMessage]


class ChatflowMessage(BaseModel):
    """Alternative message format from chatflow."""
    chatId: str
    channelId: str
    msg: str
    msgType: str = "text"
    sender: Optional[str] = None
    wa_token: Optional[str] = None
    contentUri: Optional[str] = None


class ProcessRequest(BaseModel):
    """Direct message processing request."""
    chat_id: str = Field(..., description="User's chat ID")
    sender_id: str = Field(..., description="Sender identifier")
    message: str = Field(..., description="Message text")
    source: str = Field(default="whatsapp", description="Message source")
    channel_id: str = Field(default="", description="Channel ID")


class ProcessResponse(BaseModel):
    """Message processing response."""
    response_text: str
    escalation_needed: bool
    escalation_reason: str = ""
    error: Optional[str] = None


# ============== APP SETUP ==============

app = FastAPI(
    title="Fermer Agent API",
    description="LangGraph-based AI agent for Hero's Journey customer support",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== ENDPOINTS ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "fermer-agent"}


@app.post("/webhook/wazzup", response_model=ProcessResponse)
async def wazzup_webhook(payload: WazzupWebhook, background_tasks: BackgroundTasks):
    """
    Wazzup webhook endpoint.
    
    Receives messages from Wazzup and processes them through the agent.
    Equivalent to n8n "Webhook Wazzup staryi" trigger.
    """
    if not payload.messages:
        raise HTTPException(status_code=400, detail="No messages in payload")
    
    message = payload.messages[0]
    
    # Filter: only process inbound messages
    if message.status != "inbound":
        return ProcessResponse(
            response_text="",
            escalation_needed=False,
            error="Not an inbound message"
        )
    
    # Process message
    try:
        result = await process_message(
            chat_id=message.chatId,
            sender_id=message.chatId,
            message=message.text or "",
            source=message.chatType,
            channel_id=message.channelId,
        )
        
        return ProcessResponse(
            response_text=result.get("response_text", ""),
            escalation_needed=result.get("escalation_needed", False),
            escalation_reason=result.get("escalation_reason", ""),
            error=result.get("error"),
        )
        
    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/chatflow", response_model=ProcessResponse)
async def chatflow_webhook(payload: ChatflowMessage):
    """
    Chatflow webhook endpoint.
    
    Alternative message format from chatflow system.
    Equivalent to n8n "data from chatflow" trigger.
    """
    try:
        result = await process_message(
            chat_id=payload.chatId,
            sender_id=payload.chatId,
            message=payload.msg,
            source="whatsapp",
            channel_id=payload.channelId,
        )
        
        return ProcessResponse(
            response_text=result.get("response_text", ""),
            escalation_needed=result.get("escalation_needed", False),
            escalation_reason=result.get("escalation_reason", ""),
            error=result.get("error"),
        )
        
    except Exception as e:
        logger.exception(f"Error processing chatflow message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=ProcessResponse)
async def process_direct(request: ProcessRequest):
    """
    Direct message processing endpoint.
    
    For testing or direct API integration.
    """
    try:
        result = await process_message(
            chat_id=request.chat_id,
            sender_id=request.sender_id,
            message=request.message,
            source=request.source,
            channel_id=request.channel_id,
        )
        
        return ProcessResponse(
            response_text=result.get("response_text", ""),
            escalation_needed=result.get("escalation_needed", False),
            escalation_reason=result.get("escalation_reason", ""),
            error=result.get("error"),
        )
        
    except Exception as e:
        logger.exception(f"Error processing direct message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/state/{chat_id}")
async def get_graph_state(chat_id: str):
    """
    Get current graph state for a chat.
    
    Useful for debugging and monitoring.
    """
    try:
        config = {"configurable": {"thread_id": chat_id}}
        state = await fermer_graph.aget_state(config)
        
        if state.values:
            return {
                "chat_id": chat_id,
                "trigger_type": state.values.get("trigger_type"),
                "escalation_needed": state.values.get("escalation_needed"),
                "last_response": state.values.get("humanized_response") or state.values.get("response_text"),
                "error": state.values.get("error"),
            }
        else:
            return {"chat_id": chat_id, "state": "not found"}
            
    except Exception as e:
        logger.exception(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/history/{chat_id}")
async def get_graph_history(chat_id: str, limit: int = 10):
    """
    Get conversation history for a chat.
    """
    try:
        config = {"configurable": {"thread_id": chat_id}}
        states = []
        
        async for state in fermer_graph.aget_state_history(config):
            states.append({
                "trigger_type": state.values.get("trigger_type"),
                "message": state.values.get("message"),
                "response": state.values.get("humanized_response") or state.values.get("response_text"),
            })
            if len(states) >= limit:
                break
        
        return {"chat_id": chat_id, "history": states}
        
    except Exception as e:
        logger.exception(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== MAIN ==============

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
