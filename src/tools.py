"""
Fermer Agent Tools
==================
Tools available to the AI Agent (Batyr).

Migrated from n8n:
- get_schedule_by_club (JS Code node → Python)
- Fermer vector store (Pinecone RAG)
- get_payment_link (HTTP Request node)
"""

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from typing import Optional, Literal
from datetime import datetime, timedelta
import httpx
import os


# ============== CONSTANTS ==============

GRAPHQL_ENDPOINT = "https://admin.herosjourney.kz/graphql"
AUTH_TOKEN = os.getenv("HJ_AUTH_TOKEN")

CLUB_NAMES = {
    "6788b54527af6c00ab78c66a": "Europe City",
    "67d7c4cc8b5b3112cb0bcd44": "Promenade",
    "6351ace4d61faf000b2febc8": "Nurly Orda",
    "65e9e70cbd4814536c5e27e9": "Colibri",
    "683704d8c85fb0a6b1f5a8ca": "Villa",
    "68a45233d9ba5a6ba953e5f0": "4YOU",
}

CLUB_IDS_BY_NAME = {v.lower(): k for k, v in CLUB_NAMES.items()}
CLUB_IDS_BY_NAME.update({
    "колибри": "65e9e70cbd4814536c5e27e9",
    "променад": "67d7c4cc8b5b3112cb0bcd44",
    "вилла": "683704d8c85fb0a6b1f5a8ca",
    "европа сити": "6788b54527af6c00ab78c66a",
    "нурлы орда": "6351ace4d61faf000b2febc8",
    "4ю": "68a45233d9ba5a6ba953e5f0",
})

TRAINING_TYPES = {
    "strength": {"keywords": ["RT"], "label": "силовые"},
    "bootcamp": {"keywords": ["Bootcamp"], "label": "Bootcamp"},
    "reshape": {"keywords": ["Reshape"], "label": "Reshape"},
    "assessment": {"keywords": ["Assessment"], "label": "Assessment"},
    "stretching": {"keywords": ["Stretching"], "label": "растяжка"},
    "upper": {"keywords": ["Upper"], "label": "Upper"},
    "legs": {"keywords": ["Legs"], "label": "Legs"},
    "glute": {"keywords": ["Glute"], "label": "Glute"},
    "pull": {"keywords": ["Pull"], "label": "Pull"},
    "push": {"keywords": ["Push"], "label": "Push"},
    "arm": {"keywords": ["Arm"], "label": "Arm"},
}

WEEKDAYS = {
    "monday": {"index": 0, "label": "понедельник"},
    "tuesday": {"index": 1, "label": "вторник"},
    "wednesday": {"index": 2, "label": "среда"},
    "thursday": {"index": 3, "label": "четверг"},
    "friday": {"index": 4, "label": "пятница"},
    "saturday": {"index": 5, "label": "суббота"},
    "sunday": {"index": 6, "label": "воскресенье"},
}

WEEKDAYS_RU = {
    "понедельник": 0, "вторник": 1, "среда": 2, "четверг": 3,
    "пятница": 4, "суббота": 5, "воскресенье": 6,
}

TIME_OF_DAY = {
    "morning": {"start": 6, "end": 12, "label": "утро"},
    "afternoon": {"start": 12, "end": 18, "label": "день"},
    "evening": {"start": 18, "end": 23, "label": "вечер"},
}

TIMEZONE_OFFSET = timedelta(hours=5)  # Almaty timezone


# ============== SCHEDULE TOOL ==============

@tool
async def get_schedule_by_club(
    club_id: str,
    training_type: Optional[str] = None,
    period: Optional[Literal["today", "tomorrow", "week"]] = None,
    day_of_week: Optional[str] = None,
    preferred_time: Optional[Literal["morning", "afternoon", "evening"]] = None,
) -> str:
    """
    Получает расписание тренировок Hero's Journey с фильтрацией.
    
    ПАРАМЕТРЫ:
    
    1. club_id (ОБЯЗАТЕЛЬНО) - ID клуба:
       • 65e9e70cbd4814536c5e27e9 → Colibri/Колибри
       • 67d7c4cc8b5b3112cb0bcd44 → Promenade/Променад
       • 683704d8c85fb0a6b1f5a8ca → Villa/Вилла
       • 6788b54527af6c00ab78c66a → Europe City/Европа Сити
       • 6351ace4d61faf000b2febc8 → Nurly Orda/Нурлы Орда
       • 68a45233d9ba5a6ba953e5f0 → 4YOU/4Ю
    
    2. training_type (опционально):
       • strength → все силовые (RT)
       • bootcamp → Bootcamp
       • reshape → Reshape
       • assessment → Assessment
       • stretching → растяжка
       • upper/legs/glute/pull/push/arm → детальные RT
    
    3. period (опционально): today/tomorrow/week
    
    4. day_of_week (опционально): monday-sunday (приоритет выше period)
    
    5. preferred_time (опционально): morning/afternoon/evening
    """
    # Validate club_id
    club_name = CLUB_NAMES.get(club_id)
    if not club_name:
        # Try to resolve by name
        club_id = CLUB_IDS_BY_NAME.get(club_id.lower())
        club_name = CLUB_NAMES.get(club_id) if club_id else None
        
    if not club_name:
        available = ", ".join(CLUB_NAMES.values())
        return f"❌ Укажите клуб. Доступные: {available}"
    
    # Calculate date range
    now = datetime.utcnow() + TIMEZONE_OFFSET
    
    # Get week range (Monday to Sunday)
    days_since_monday = now.weekday()
    monday = now - timedelta(days=days_since_monday)
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    sunday = monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    # Convert back to UTC for API
    start_time = (monday - TIMEZONE_OFFSET).isoformat() + "Z"
    end_time = (sunday - TIMEZONE_OFFSET).isoformat() + "Z"
    
    # GraphQL query
    query = """
    query EventsByDates($startTime: String!, $endTime: String!, $clubId: String!) {
        eventsByDates(startTime: $startTime, endTime: $endTime, clubId: $clubId) {
            id
            startTime
            endTime
            status
            programSet {
                name
            }
        }
    }
    """
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GRAPHQL_ENDPOINT,
                json={
                    "query": query,
                    "variables": {
                        "startTime": start_time,
                        "endTime": end_time,
                        "clubId": club_id,
                    }
                },
                headers={
                    "Authorization": f"Bearer {AUTH_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            data = response.json()
    except Exception as e:
        return f"⚠️ Ошибка получения расписания: {str(e)}"
    
    events = data.get("data", {}).get("eventsByDates", [])
    if not events:
        return f"📅 В {club_name} нет запланированных тренировок."
    
    # Filter by status
    trainings = [
        e for e in events
        if e.get("status") != "finished"
        and "[TEST]" not in (e.get("programSet", {}).get("name") or "")
    ]
    
    # Filter by period/day
    if day_of_week and day_of_week.lower() in WEEKDAYS:
        target_day = WEEKDAYS[day_of_week.lower()]["index"]
        trainings = [
            t for t in trainings
            if _parse_datetime(t["startTime"]).weekday() == target_day
        ]
    elif period == "today":
        today = now.date()
        trainings = [
            t for t in trainings
            if _parse_datetime(t["startTime"]).date() == today
        ]
    elif period == "tomorrow":
        tomorrow = (now + timedelta(days=1)).date()
        trainings = [
            t for t in trainings
            if _parse_datetime(t["startTime"]).date() == tomorrow
        ]
    
    # Filter by training type
    if training_type and training_type in TRAINING_TYPES:
        keywords = TRAINING_TYPES[training_type]["keywords"]
        trainings = [
            t for t in trainings
            if any(kw in (t.get("programSet", {}).get("name") or "") for kw in keywords)
        ]
    
    # Filter by time of day
    if preferred_time and preferred_time in TIME_OF_DAY:
        time_range = TIME_OF_DAY[preferred_time]
        trainings = [
            t for t in trainings
            if time_range["start"] <= _parse_datetime(t["startTime"]).hour < time_range["end"]
        ]
    
    if not trainings:
        filter_desc = _build_filter_description(period, day_of_week, training_type, preferred_time)
        return f"📅 В {club_name} {filter_desc} нет подходящих тренировок. Попробуйте изменить фильтры."
    
    # Format output
    return _format_schedule(trainings, club_name, club_id, period, day_of_week, training_type, preferred_time)


def _parse_datetime(iso_string: str) -> datetime:
    """Parse ISO datetime string and convert to local time."""
    dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    return dt.replace(tzinfo=None) + TIMEZONE_OFFSET


def _build_filter_description(period, day_of_week, training_type, preferred_time) -> str:
    """Build human-readable filter description."""
    parts = []
    
    if day_of_week and day_of_week.lower() in WEEKDAYS:
        parts.append(f"на {WEEKDAYS[day_of_week.lower()]['label']}")
    elif period == "today":
        parts.append("на сегодня")
    elif period == "tomorrow":
        parts.append("на завтра")
    else:
        parts.append("на неделю")
    
    if training_type and training_type in TRAINING_TYPES:
        parts.append(f"| {TRAINING_TYPES[training_type]['label']}")
    
    if preferred_time and preferred_time in TIME_OF_DAY:
        parts.append(f"| {TIME_OF_DAY[preferred_time]['label']}")
    
    return " ".join(parts)


def _format_schedule(trainings, club_name, club_id, period, day_of_week, training_type, preferred_time) -> str:
    """Format schedule for display."""
    filter_desc = _build_filter_description(period, day_of_week, training_type, preferred_time)
    
    # Group by date
    by_date = {}
    for t in trainings:
        dt = _parse_datetime(t["startTime"])
        date_key = dt.strftime("%Y-%m-%d")
        if date_key not in by_date:
            by_date[date_key] = {
                "display": _format_date(dt),
                "trainings": [],
            }
        by_date[date_key]["trainings"].append({
            "time": dt.strftime("%H:%M"),
            "name": t.get("programSet", {}).get("name") or "Тренировка",
            "id": t["id"],
            "dt": dt,
        })
    
    # Sort
    sorted_dates = sorted(by_date.keys())
    
    lines = [f"📅 {club_name} {filter_desc}:\n"]
    
    for date_key in sorted_dates:
        day = by_date[date_key]
        day["trainings"].sort(key=lambda x: x["dt"])
        
        lines.append(f"\n📆 {day['display']}")
        
        for t in day["trainings"]:
            lines.append(f"  🕐 {t['time']} | {t['name']} [id:{t['id']}]")
    
    lines.append(f"\n📋 Для записи: используй eventId из [id:...] и clubId: {club_id}")
    
    return "\n".join(lines)


def _format_date(dt: datetime) -> str:
    """Format date in Russian."""
    days = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье']
    months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
              'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
    return f"{days[dt.weekday()]}, {dt.day} {months[dt.month - 1]}"


# ============== KNOWLEDGE BASE TOOL (Pinecone RAG) ==============

# Lazy initialization for Pinecone and embeddings
_pc = None
_embeddings = None


def _get_pinecone():
    """Lazy initialization of Pinecone client."""
    global _pc
    if _pc is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        _pc = Pinecone(api_key=api_key)
    return _pc


def _get_embeddings():
    """Lazy initialization of OpenAI embeddings."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return _embeddings


@tool
async def search_knowledge_base(query: str) -> str:
    """
    🔴 ОБЯЗАТЕЛЬНАЯ БАЗА ЗНАНИЙ — ИЩИ ПЕРЕД КАЖДЫМ ОТВЕТОМ
    
    Содержит ВСЮ актуальную информацию о Hero's Journey:
    - Цены и условия абонементов
    - Программы тренировок  
    - Правила студии
    - ГОТОВЫЕ СКРИПТЫ отработки возражений
    
    ⚠️ ПРАВИЛА:
    1. НИКОГДА не отвечай из памяти — ВСЕГДА ищи сначала
    2. При возражении клиента — ОБЯЗАТЕЛЬНО ищи готовый скрипт
    3. Адаптируй найденный ответ под контекст диалога
    
    🔍 КОГДА ИСКАТЬ:
    
    ЦЕНОВЫЕ ВОЗРАЖЕНИЯ:
    - "дорого", "слишком дорого" → ищи: "возражение дорого цена"
    - "нет денег" → ищи: "возражение нет денег бюджет"
    - "в другом зале дешевле" → ищи: "возражение другой зал дешевле конкурент"
    
    ОТКЛАДЫВАНИЕ:
    - "подумаю", "надо подумать" → ищи: "возражение подумаю"
    - "с понедельника", "после праздников" → ищи: "возражение понедельник откладывание"
    
    ВРЕМЕННЫЕ:
    - "нет времени" → ищи: "возражение нет времени занят"
    - "далеко ездить" → ищи: "возражение далеко локация клуб"
    
    ЦЕНЫ И ПРОДУКТЫ:
    - Hero's Pass цена → ищи: "Hero's Pass цена стоимость"
    - рассрочка → ищи: "рассрочка 0-0-12 Kaspi"
    - First Step/Basecamp/Hero's Week → ищи: "trial программа First Step Basecamp"
    
    Args:
        query: Поисковый запрос на русском языке
    
    Returns:
        Релевантные документы из базы знаний
    """
    try:
        pc = _get_pinecone()
        embeddings = _get_embeddings()
        index = pc.Index("fermer-knowledge")

        # Get embedding for query
        query_embedding = await embeddings.aembed_query(query)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            namespace="knowledge_base",
            include_metadata=True,
        )
        
        if not results.matches:
            return "Информация не найдена в базе знаний. Попробуйте другой запрос."
        
        # Format results
        docs = []
        for match in results.matches:
            score = match.score
            text = match.metadata.get("text", "")
            source = match.metadata.get("source", "")
            
            if score > 0.7:  # Only include relevant results
                docs.append(f"[Релевантность: {score:.0%}]\n{text}")
        
        if not docs:
            return "Релевантных документов не найдено. Попробуйте уточнить запрос."
        
        return "\n\n---\n\n".join(docs)
        
    except Exception as e:
        return f"Ошибка поиска в базе знаний: {str(e)}"


# ============== PAYMENT LINK TOOL ==============

@tool
async def get_payment_link(
    product: Literal["heros_week", "basecamp", "first_step", "heros_pass_6", "heros_pass_12"],
    club_id: str,
    chat_id: str,
) -> str:
    """
    Генерирует ссылку на оплату продукта Hero's Journey.
    
    ПРОДУКТЫ:
    - heros_week: Hero's Week (9 990 ₸) - 1 неделя trial
    - basecamp: Basecamp (29 990 ₸) - 2 недели trial
    - first_step: Первый Шаг (59 990 ₸) - 1 месяц trial
    - heros_pass_6: Hero's Pass 6 месяцев (349 990 ₸)
    - heros_pass_12: Hero's Pass 12 месяцев (549 990 ₸)
    
    Args:
        product: Код продукта
        club_id: ID клуба
        chat_id: ID чата клиента для привязки платежа
    
    Returns:
        Ссылка на оплату или сообщение об ошибке
    """
    PRODUCT_CONFIGS = {
        "heros_week": {"name": "Hero's Week", "price": 9990},
        "basecamp": {"name": "Basecamp", "price": 29990},
        "first_step": {"name": "Первый Шаг", "price": 59990},
        "heros_pass_6": {"name": "Hero's Pass 6 мес", "price": 349990},
        "heros_pass_12": {"name": "Hero's Pass 12 мес", "price": 549990},
    }
    
    if product not in PRODUCT_CONFIGS:
        return f"❌ Неизвестный продукт. Доступные: {', '.join(PRODUCT_CONFIGS.keys())}"
    
    config = PRODUCT_CONFIGS[product]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GRAPHQL_ENDPOINT}/payment/create-link",
                json={
                    "product": product,
                    "clubId": club_id,
                    "chatId": chat_id,
                    "amount": config["price"],
                },
                headers={
                    "Authorization": f"Bearer {AUTH_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            data = response.json()
            
        if "paymentUrl" in data:
            return f"✅ Ссылка на оплату {config['name']} ({config['price']:,} ₸):\n{data['paymentUrl']}"
        else:
            return f"⚠️ Не удалось создать ссылку на оплату. Попросите клиента связаться с менеджером."
            
    except Exception as e:
        return f"⚠️ Ошибка создания ссылки: {str(e)}"


# ============== EXPORT ==============

__all__ = [
    "get_schedule_by_club",
    "search_knowledge_base", 
    "get_payment_link",
]
