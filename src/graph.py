"""
Fermer Respond Agent - LangGraph Migration
==========================================
Миграция n8n workflow "Fermer respond agent" на LangGraph.

Архитектура:
    Webhook → Extract Data → Get Fermer Data → Select Prompt (by trigger)
        → AI Agent (Batyr) + Tools → Humanizer → Send Response
        → If escalation → Notify Managers
"""

from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
import re
from datetime import datetime

from tools import get_schedule_by_club, search_knowledge_base, get_payment_link
from prompts import get_system_prompt, get_user_prompt
from integrations import (
    fetch_fermer_data,
    log_message_to_db,
    send_whatsapp_message,
    notify_telegram,
    update_amocrm_lead
)


# ============== STATE DEFINITION ==============

class EscalationInfo(BaseModel):
    """Structured output for escalation decisions"""
    needed: bool = Field(description="Whether human escalation is needed")
    reason: str = Field(default="", description="Reason for escalation if needed")


class AgentResponse(BaseModel):
    """Structured output from AI agent"""
    response_text: str = Field(description="Response message to customer in Russian")
    escalation: EscalationInfo = Field(description="Escalation information")


class FermerState(TypedDict):
    """
    State schema for Fermer Agent workflow.
    
    Mirrors n8n workflow data flow:
    - Input from webhook
    - Fermer data from GraphQL
    - Trigger-based prompt selection
    - AI response with escalation flag
    - Output actions
    """
    # Input data (from webhook)
    chat_id: str
    sender_id: str
    message: str
    source: str  # whatsapp, telegram, etc.
    channel_id: str
    timestamp: str
    
    # Fermer data (from GraphQL)
    user_id: Optional[str]
    query_id: Optional[str]
    user_data: Optional[dict]
    user_profile: Optional[dict]
    messages_history: list[str]
    training_data: Optional[dict]
    triggers: dict  # first_training, no_activity, finish_program, payment
    club_id: Optional[str]
    club_manager_tg: Optional[int]
    club_tg_chat: Optional[int]
    
    # Prompt selection
    trigger_type: str  # first_training, no_activity, finish_program, payment, default
    system_prompt: str
    user_prompt: str
    
    # AI conversation
    messages: Annotated[list, add_messages]
    
    # Agent output
    response_text: str
    humanized_response: str
    escalation_needed: bool
    escalation_reason: str
    
    # Workflow control
    error: Optional[str]
    should_respond: bool


# ============== CLUB MAPPINGS ==============

CLUB_MANAGERS = {
    "6351ace4d61faf000b2febc8": 8800966,   # Нурлы Орда → Камиля
    "65e9e70cbd4814536c5e27e9": 10738998,  # Colibri → Рысалды
    "6788b54527af6c00ab78c66a": 11613982,  # Europe City → Дана
    "683704d8c85fb0a6b1f5a8ca": 12536974,  # Villa → Салтанат
    "67d7c4cc8b5b3112cb0bcd44": 12234034,  # Promenade → Аида
    "68a45233d9ba5a6ba953e5f0": 12885486,  # 4you → Мадина
}

CLUB_TG_CHATS = {
    "65e9e70cbd4814536c5e27e9": -4900775642,     # Colibri
    "6351ace4d61faf000b2febc8": -1002535386890,  # Nurly Orda
    "683704d8c85fb0a6b1f5a8ca": -1002648405729,  # Villa
    "67d7c4cc8b5b3112cb0bcd44": -1002765678928,  # Promenade
    "6788b54527af6c00ab78c66a": -1002664385193,  # Europe City
    "68a45233d9ba5a6ba953e5f0": -1003568350790,  # 4YOU
}


# ============== NODE FUNCTIONS ==============

def extract_message_data(state: FermerState) -> dict:
    """
    Node 1: Extract and normalize message data from webhook.
    Equivalent to n8n "Extract Message Data" node.
    """
    return {
        "timestamp": state.get("timestamp") or datetime.now().isoformat(),
        "should_respond": bool(state.get("message")),
    }


async def fetch_fermer_data_node(state: FermerState) -> dict:
    """
    Node 2: Fetch fermer data from GraphQL.
    Equivalent to n8n "get fermer data" + "Simplify data" nodes.
    """
    try:
        fermer_data = await fetch_fermer_data(state["sender_id"])
        
        if not fermer_data:
            return {
                "error": "No fermer data found",
                "should_respond": False,
            }
        
        # Extract user and profile data
        user_data = fermer_data.get("user", {})
        user_profile = fermer_data.get("userProfile", {})
        club_id = user_data.get("club", {}).get("id")
        
        # Get current query and dialog history
        queries = fermer_data.get("queries", [])
        current_query = queries[-1] if queries else {}
        
        # Build messages history
        dialog = current_query.get("dialog", [])
        messages_history = [
            f"{msg.get('sender', '')} ({msg.get('created_at', '')}): {msg.get('text', '')}"
            for msg in dialog
        ]
        
        # Extract training data from last auto message
        training_data = {}
        auto_messages = [m for m in dialog if m.get("sender") == "auto"]
        if auto_messages and auto_messages[-1].get("trainingData"):
            training_data = auto_messages[-1]["trainingData"]
        
        # Get triggers
        triggers = fermer_data.get("triggers", {
            "payment": True,
            "firstTraining": True,
            "noActivity": True,
            "finishProgram": True,
        })
        
        return {
            "user_id": fermer_data.get("userId"),
            "query_id": current_query.get("id"),
            "user_data": user_data,
            "user_profile": user_profile,
            "messages_history": messages_history,
            "training_data": training_data,
            "triggers": triggers,
            "club_id": club_id,
            "club_manager_tg": CLUB_MANAGERS.get(club_id),
            "club_tg_chat": CLUB_TG_CHATS.get(club_id),
        }
        
    except Exception as e:
        return {
            "error": f"Failed to fetch fermer data: {str(e)}",
            "should_respond": False,
        }


def select_trigger_type(state: FermerState) -> dict:
    """
    Node 3: Determine trigger type based on user state.
    Equivalent to n8n "Switch" node.
    """
    triggers = state.get("triggers", {})
    
    if triggers.get("firstTraining"):
        trigger_type = "first_training"
    elif triggers.get("noActivity"):
        trigger_type = "no_activity"
    elif triggers.get("finishProgram"):
        trigger_type = "finish_program"
    elif triggers.get("payment"):
        trigger_type = "payment"
    else:
        trigger_type = "default"
    
    return {"trigger_type": trigger_type}


def build_prompts(state: FermerState) -> dict:
    """
    Node 4: Build system and user prompts based on trigger type.
    Equivalent to n8n "set first training prompts", "set no activity prompt", etc.
    """
    trigger_type = state["trigger_type"]
    
    system_prompt = get_system_prompt(
        trigger_type=trigger_type,
        user_data=state.get("user_data", {}),
        user_profile=state.get("user_profile", {}),
    )
    
    user_prompt = get_user_prompt(
        trigger_type=trigger_type,
        message=state["message"],
        messages_history=state.get("messages_history", []),
        training_data=state.get("training_data", {}),
        user_data=state.get("user_data", {}),
    )
    
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
    }


async def ai_agent_node(state: FermerState) -> dict:
    """
    Node 5: Main AI Agent (Batyr) with tools.
    Equivalent to n8n "AI agent RAG" node.

    Tools available:
    - get_schedule_by_club: Fetch training schedule
    - search_knowledge_base: RAG search in Pinecone
    - get_payment_link: Generate payment link

    This implements a ReAct-style agent that:
    1. Calls tools as needed
    2. Iterates until the LLM produces a final response
    3. Parses the JSON output for escalation info
    """
    tools = [get_schedule_by_club, search_knowledge_base, get_payment_link]
    tools_by_name = {tool.name: tool for tool in tools}

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
    ).bind_tools(tools)

    messages = list(state["messages"])
    max_iterations = 5  # Prevent infinite loops

    try:
        for _ in range(max_iterations):
            # Call LLM
            response = await llm.ainvoke(messages)
            messages.append(response)

            # Check if LLM wants to call tools
            if response.tool_calls:
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name in tools_by_name:
                        tool = tools_by_name[tool_name]
                        # Execute the tool
                        try:
                            tool_result = await tool.ainvoke(tool_args)
                        except Exception as e:
                            tool_result = f"Error executing {tool_name}: {str(e)}"

                        # Add tool result to messages
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"],
                        ))
                    else:
                        messages.append(ToolMessage(
                            content=f"Unknown tool: {tool_name}",
                            tool_call_id=tool_call["id"],
                        ))
            else:
                # No tool calls - LLM produced final response
                break

        # Parse the final response for escalation info
        final_content = response.content
        escalation_needed = False
        escalation_reason = ""
        response_text = final_content

        # Try to parse as JSON (as specified in system prompt)
        try:
            # Extract JSON from response if wrapped in markdown
            json_match = re.search(r'\{[\s\S]*\}', final_content)
            if json_match:
                parsed = json.loads(json_match.group())
                response_text = parsed.get("response", parsed.get("response_text", final_content))
                escalation_data = parsed.get("escalation", {})
                if isinstance(escalation_data, dict):
                    escalation_needed = escalation_data.get("needed", False)
                    escalation_reason = escalation_data.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            # If not valid JSON, use the raw response
            pass

        return {
            "response_text": response_text,
            "escalation_needed": escalation_needed,
            "escalation_reason": escalation_reason,
            "messages": messages,
        }

    except Exception as e:
        return {
            "error": f"AI agent error: {str(e)}",
            "should_respond": False,
        }


async def humanizer_node(state: FermerState) -> dict:
    """
    Node 6: Humanize AI response.
    Equivalent to n8n "Huminize Agent" node.
    
    Makes responses more natural and human-like while:
    - Preserving all facts and data
    - Adapting tone to time of day
    - Matching user's emotional state
    """
    if not state.get("response_text"):
        return {"humanized_response": ""}
    
    humanizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    
    # Get current hour for time-based adaptation
    current_hour = datetime.now().hour
    if 6 <= current_hour < 11:
        time_context = "утро - энергичный тон"
    elif 11 <= current_hour < 17:
        time_context = "день - деловой тон"
    elif 17 <= current_hour < 23:
        time_context = "вечер - спокойный тон"
    else:
        time_context = "ночь - лаконичный тон"
    
    humanizer_prompt = f"""
Ты — Humanizer Agent. Твоя задача — сделать AI-ответ более живым и естественным.

ПРАВИЛА:
1. Сохрани ВСЮ информацию (цены, даты, имена)
2. Время суток: {time_context}
3. Используй "Вы" (формально)
4. Не добавляй новой информации
5. Макс 600 символов
6. 1-3 коротких абзаца

ЗАПРЕЩЕНО:
- "Уважаемый", "С уважением"
- Канцеляризмы
- Добавлять то, чего нет в оригинале

Оригинальный ответ:
{state["response_text"]}

Гуманизированная версия:
"""
    
    try:
        result = await humanizer_llm.ainvoke([HumanMessage(content=humanizer_prompt)])
        return {"humanized_response": result.content}
    except Exception:
        # Fallback to original response
        return {"humanized_response": state["response_text"]}


async def send_response_node(state: FermerState) -> dict:
    """
    Node 7: Send response via WhatsApp.
    Equivalent to n8n "Отправить сообщение" node.
    """
    response = state.get("humanized_response") or state.get("response_text", "")
    
    if not response or not state.get("should_respond", True):
        return {}
    
    try:
        await send_whatsapp_message(
            chat_id=state["sender_id"],
            channel_id=state["channel_id"],
            text=response,
            source=state.get("source", "whatsapp"),
        )
        
        # Log AI message to database
        await log_message_to_db(
            query_id=state.get("query_id"),
            chat_id=state["chat_id"],
            user_id=state.get("user_id"),
            text=response,
            sender="ai",
        )
        
        return {}
        
    except Exception as e:
        return {"error": f"Failed to send response: {str(e)}"}


async def handle_escalation_node(state: FermerState) -> dict:
    """
    Node 8: Handle escalation to human managers.
    Equivalent to n8n "If Human Needed" → Telegram/Notion flow.
    """
    if not state.get("escalation_needed"):
        return {}
    
    try:
        # Notify Telegram group
        tg_chat = state.get("club_tg_chat") or -1003234914487  # Default managers group
        
        await notify_telegram(
            chat_id=tg_chat,
            message=f"🚨 Эскалация: {state['sender_id']}\n\n"
                    f"Причина: {state.get('escalation_reason', 'Не указана')}\n\n"
                    f"Последнее сообщение:\n"
                    f"USER: {state['message']}\n"
                    f"AI: {state.get('response_text', '')}",
        )
        
        # Update AmoCRM lead status
        await update_amocrm_lead(
            chat_id=state["sender_id"],
            status="human_needed",
        )
        
        return {}
        
    except Exception as e:
        return {"error": f"Escalation failed: {str(e)}"}


# ============== CONDITIONAL EDGES ==============

def should_continue_after_data(state: FermerState) -> Literal["select_trigger", "end"]:
    """Check if we have valid fermer data to continue."""
    if state.get("error") or not state.get("should_respond", True):
        return "end"
    return "select_trigger"


def should_escalate(state: FermerState) -> Literal["escalate", "end"]:
    """Check if escalation is needed."""
    if state.get("escalation_needed"):
        return "escalate"
    return "end"


# ============== BUILD GRAPH ==============

def create_fermer_graph():
    """
    Build the Fermer Agent LangGraph workflow.
    
    Graph structure:
    
    START → extract_data → fetch_fermer_data → [check data]
        → select_trigger → build_prompts → ai_agent → humanizer → send_response
            → [check escalation] → handle_escalation → END
    """
    workflow = StateGraph(FermerState)
    
    # Add nodes
    workflow.add_node("extract_data", extract_message_data)
    workflow.add_node("fetch_fermer_data", fetch_fermer_data_node)
    workflow.add_node("select_trigger", select_trigger_type)
    workflow.add_node("build_prompts", build_prompts)
    workflow.add_node("ai_agent", ai_agent_node)
    workflow.add_node("humanizer", humanizer_node)
    workflow.add_node("send_response", send_response_node)
    workflow.add_node("handle_escalation", handle_escalation_node)
    
    # Add edges
    workflow.add_edge(START, "extract_data")
    workflow.add_edge("extract_data", "fetch_fermer_data")
    
    workflow.add_conditional_edges(
        "fetch_fermer_data",
        should_continue_after_data,
        {
            "select_trigger": "select_trigger",
            "end": END,
        }
    )
    
    workflow.add_edge("select_trigger", "build_prompts")
    workflow.add_edge("build_prompts", "ai_agent")
    workflow.add_edge("ai_agent", "humanizer")
    workflow.add_edge("humanizer", "send_response")
    
    workflow.add_conditional_edges(
        "send_response",
        should_escalate,
        {
            "escalate": "handle_escalation",
            "end": END,
        }
    )
    
    workflow.add_edge("handle_escalation", END)
    
    # Compile with checkpointer for state persistence
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Create graph instance
fermer_graph = create_fermer_graph()


# ============== ENTRY POINT ==============

async def process_message(
    chat_id: str,
    sender_id: str,
    message: str,
    source: str = "whatsapp",
    channel_id: str = "",
) -> dict:
    """
    Main entry point for processing incoming messages.
    
    Args:
        chat_id: Unique chat identifier
        sender_id: User's phone/ID
        message: Message text
        source: Message source (whatsapp, telegram)
        channel_id: Wazzup channel ID
    
    Returns:
        Final state with response and escalation info
    """
    initial_state = {
        "chat_id": chat_id,
        "sender_id": sender_id,
        "message": message,
        "source": source,
        "channel_id": channel_id,
        "timestamp": datetime.now().isoformat(),
        "messages": [],
        "messages_history": [],
        "triggers": {},
        "should_respond": True,
    }
    
    config = {"configurable": {"thread_id": chat_id}}
    
    result = await fermer_graph.ainvoke(initial_state, config)
    
    return {
        "response_text": result.get("humanized_response") or result.get("response_text", ""),
        "escalation_needed": result.get("escalation_needed", False),
        "escalation_reason": result.get("escalation_reason", ""),
        "error": result.get("error"),
    }
