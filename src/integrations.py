"""
Fermer Agent Integrations
=========================
External service integrations.

Migrated from n8n:
- GraphQL API (Hero's Journey backend)
- Wazzup API (WhatsApp messaging)
- Telegram Bot API
- AmoCRM API
"""

import httpx
import os
from typing import Optional, Literal


# ============== CONFIGURATION ==============

GRAPHQL_ENDPOINT = "https://admin.herosjourney.kz/graphql"
HJ_AUTH_TOKEN = os.getenv("HJ_AUTH_TOKEN")

WAZZUP_API = "https://api.wazzup24.com/v3"
WAZZUP_TOKEN = os.getenv("WAZZUP_TOKEN")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

AMOCRM_DOMAIN = "fitnesslabs123.amocrm.ru"
AMOCRM_TOKEN = os.getenv("AMOCRM_TOKEN")
AMOCRM_PIPELINE_ID = "10354830"
AMOCRM_STATUS_INITIAL = "81882938"
AMOCRM_STATUS_HUMAN = "81914526"

# Notion configuration
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = "29b0a16f-4371-81cc-9794-ce306f1d13c6"  # From n8n workflow


# ============== GRAPHQL INTEGRATION ==============

async def fetch_fermer_data(chat_id: str) -> Optional[dict]:
    """
    Fetch fermer data from Hero's Journey GraphQL API.
    
    Equivalent to n8n "get fermer data" node.
    
    Args:
        chat_id: User's chat ID (phone number)
    
    Returns:
        Fermer data dict or None if not found
    """
    query = """
    query FermerByChatId($chatId: String!) {
        fermerByChatId(chatId: $chatId) {
            id
            userId
            marathonEndDate
            chatId
            created_at
            updated_at
            triggers {
                payment
                firstTraining
                noActivity
                finishProgram
                notBuy
            }
            queries {
                id
                adminId
                created_at
                dialog {
                    text
                    sender
                    triggerType
                    created_at
                    trainingData {
                        firstName
                        phoneNumber
                        bookingId
                        eventId
                        eventName
                        eventEndTime
                        hasCheckedIn
                        birthdayDate
                        userSex
                        heartRateData {
                            max_hr
                            average_hr
                            calories
                            dumbbells
                            weight
                            place
                            zone_duration {
                                Zone0
                                Zone1
                                Zone2
                                Zone3
                                Zone4
                                Zone5
                            }
                        }
                        eventRating {
                            id
                            value
                            comment
                            ratingByEvent
                            ratingByTrainer
                            commentByEvent
                            commentByTrainer
                        }
                        lastMarathon {
                            marathonId
                            marathonName
                            startTime
                            endTime
                            status
                        }
                        lastPayment {
                            paymentId
                            amount
                            currency
                            status
                            created_at
                            product
                        }
                        activeMarathon {
                            marathonId
                            marathonName
                            startTime
                            endTime
                            status
                        }
                        totalWeight
                        trainingCount
                        totalCalories
                        avgRatingByEvent
                        avgRatingByTrainer
                        allRatings
                    }
                }
            }
            user {
                firstName
                lastName
                sex
                tickets
                club {
                    id
                    name
                }
                age
                weight
                height
            }
            userProfile {
                goal
                fitnessLevel
                timePreferences
                healthLimitations
                barriers
                sourceAttraction
                previousGymsExperience
                motivation_type
                communication_style
                preferred_language
                objections_mentioned
                interests
                sentiment_overall
                escalated_to_human
                escalation_reason
                funnel_stage
                confidence_score
                lead_temperature
                last_stage_change_at
                collectedAt
                completeness
            }
        }
    }
    """
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GRAPHQL_ENDPOINT,
            json={
                "query": query,
                "variables": {"chatId": chat_id}
            },
            headers={
                "Authorization": f"Bearer {HJ_AUTH_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        
        data = response.json()
        return data.get("data", {}).get("fermerByChatId")


async def log_message_to_db(
    query_id: str,
    chat_id: str,
    user_id: str,
    text: str,
    sender: Literal["user", "ai", "auto"],
) -> bool:
    """
    Log message to Hero's Journey database.
    
    Equivalent to n8n "log user message" node.
    
    Args:
        query_id: Current query ID
        chat_id: User's chat ID
        user_id: User's ID in the system
        text: Message text
        sender: Message sender type
    
    Returns:
        True if successful
    """
    # Escape special characters for GraphQL
    escaped_text = (
        text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
    )
    
    mutation = f"""
    mutation AddFermerMessage {{
        addFermerMessage(
            queryId: "{query_id}"
            message: {{ text: "{escaped_text}", sender: "{sender}" }}
            chatId: "{chat_id}"
            userId: "{user_id}"
        ) {{
            id
            userId
            chatId
        }}
    }}
    """
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GRAPHQL_ENDPOINT,
            json={"query": mutation},
            headers={
                "Authorization": f"Bearer {HJ_AUTH_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        
        data = response.json()
        return "errors" not in data


# ============== WAZZUP INTEGRATION ==============

async def send_whatsapp_message(
    chat_id: str,
    channel_id: str,
    text: str,
    source: str = "whatsapp",
) -> bool:
    """
    Send message via Wazzup API.
    
    Equivalent to n8n "Отправить сообщение" node.
    
    Args:
        chat_id: Recipient chat ID
        channel_id: Wazzup channel ID
        text: Message text
        source: Chat type (whatsapp, telegram)
    
    Returns:
        True if successful
    """
    import time
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{WAZZUP_API}/message",
            json={
                "channelId": channel_id,
                "chatType": source,
                "crmMessageId": f"msg-{int(time.time() * 1000)}",
                "chatId": chat_id,
                "text": text,
            },
            headers={
                "Authorization": f"Bearer {WAZZUP_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        
        return response.status_code == 200


# ============== TELEGRAM INTEGRATION ==============

async def notify_telegram(
    chat_id: int,
    message: str,
) -> bool:
    """
    Send notification to Telegram chat.
    
    Equivalent to n8n "Send a text message" Telegram node.
    
    Args:
        chat_id: Telegram chat ID (group or user)
        message: Message text
    
    Returns:
        True if successful
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=30.0,
        )
        
        return response.status_code == 200


# ============== AMOCRM INTEGRATION ==============

async def get_lead_by_chat_id(chat_id: str) -> Optional[dict]:
    """
    Find AmoCRM lead by chat ID.
    
    Args:
        chat_id: User's chat ID
    
    Returns:
        Lead data or None
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://{AMOCRM_DOMAIN}/api/v4/leads",
            params={
                "query": chat_id,
                "filter[pipeline_id]": AMOCRM_PIPELINE_ID,
            },
            headers={
                "Authorization": f"Bearer {AMOCRM_TOKEN}",
            },
            timeout=30.0,
        )
        
        data = response.json()
        leads = data.get("_embedded", {}).get("leads", [])
        return leads[0] if leads else None


async def update_amocrm_lead(
    chat_id: str,
    status: Literal["initial", "human_needed"],
) -> bool:
    """
    Update AmoCRM lead status.
    
    Equivalent to n8n AmoCRM status update nodes.
    
    Args:
        chat_id: User's chat ID
        status: New status (initial or human_needed)
    
    Returns:
        True if successful
    """
    # Get existing lead
    lead = await get_lead_by_chat_id(chat_id)
    
    status_id = AMOCRM_STATUS_INITIAL if status == "initial" else AMOCRM_STATUS_HUMAN
    
    if lead:
        # Update existing lead
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"https://{AMOCRM_DOMAIN}/api/v4/leads/{lead['id']}",
                json={
                    "status_id": int(status_id),
                },
                headers={
                    "Authorization": f"Bearer {AMOCRM_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            return response.status_code == 200
    else:
        # Create new lead
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://{AMOCRM_DOMAIN}/api/v4/leads",
                json=[{
                    "name": f"Fermer Lead {chat_id}",
                    "pipeline_id": int(AMOCRM_PIPELINE_ID),
                    "status_id": int(status_id),
                    "custom_fields_values": [
                        {
                            "field_id": 3031325,  # custom_chatid_field_id
                            "values": [{"value": chat_id}]
                        }
                    ]
                }],
                headers={
                    "Authorization": f"Bearer {AMOCRM_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            return response.status_code == 200


# ============== NOTION INTEGRATION ==============

async def create_notion_escalation(
    chat_id: str,
    user_name: str,
    escalation_reason: str,
    last_message: str,
    ai_response: str,
    club_name: str = "",
) -> bool:
    """
    Create escalation page in Notion database.

    Equivalent to n8n "Create a database page2" node.

    Args:
        chat_id: User's chat ID
        user_name: User's name
        escalation_reason: Why escalation was triggered
        last_message: Last user message
        ai_response: AI's response before escalation
        club_name: Club name

    Returns:
        True if successful
    """
    if not NOTION_TOKEN:
        return False

    from datetime import datetime

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.notion.com/v1/pages",
            json={
                "parent": {"database_id": NOTION_DATABASE_ID},
                "properties": {
                    "Name": {
                        "title": [
                            {"text": {"content": f"Escalation: {user_name} ({chat_id})"}}
                        ]
                    },
                    "Chat ID": {
                        "rich_text": [
                            {"text": {"content": chat_id}}
                        ]
                    },
                    "Reason": {
                        "rich_text": [
                            {"text": {"content": escalation_reason}}
                        ]
                    },
                    "Club": {
                        "rich_text": [
                            {"text": {"content": club_name}}
                        ]
                    },
                    "Status": {
                        "select": {"name": "New"}
                    },
                    "Created": {
                        "date": {"start": datetime.now().isoformat()}
                    },
                },
                "children": [
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [{"text": {"content": "Last User Message"}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"text": {"content": last_message}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [{"text": {"content": "AI Response"}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"text": {"content": ai_response}}]
                        }
                    },
                ]
            },
            headers={
                "Authorization": f"Bearer {NOTION_TOKEN}",
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28",
            },
            timeout=30.0,
        )

        return response.status_code == 200


# ============== EXPORT ==============

__all__ = [
    "fetch_fermer_data",
    "log_message_to_db",
    "send_whatsapp_message",
    "notify_telegram",
    "update_amocrm_lead",
    "create_notion_escalation",
]
