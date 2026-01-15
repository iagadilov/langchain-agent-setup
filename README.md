# Fermer Agent — LangGraph Migration

AI-агент для Hero's Journey, мигрированный с n8n на LangGraph.

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Fermer Agent Graph                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  START                                                              │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────┐                                                │
│  │  extract_data   │  ← Webhook input                               │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────────┐                                            │
│  │  fetch_fermer_data  │  ← GraphQL: user data, history, triggers   │
│  └────────┬────────────┘                                            │
│           │                                                         │
│           ▼                                                         │
│     ┌─────┴─────┐                                                   │
│     │  check    │                                                   │
│     │  data     │                                                   │
│     └─────┬─────┘                                                   │
│       ┌───┴───┐                                                     │
│    error?    ok                                                     │
│       │       │                                                     │
│       ▼       ▼                                                     │
│      END   ┌──────────────────┐                                     │
│            │  select_trigger  │  ← first_training / no_activity /   │
│            └────────┬─────────┘    finish_program / payment         │
│                     │                                               │
│                     ▼                                               │
│            ┌─────────────────┐                                      │
│            │  build_prompts  │  ← Dynamic system + user prompts     │
│            └────────┬────────┘                                      │
│                     │                                               │
│                     ▼                                               │
│            ┌─────────────────┐   ┌──────────────────────────┐       │
│            │    ai_agent     │───│  Tools:                  │       │
│            │    (Batyr)      │   │  • get_schedule_by_club  │       │
│            └────────┬────────┘   │  • search_knowledge_base │       │
│                     │            │  • get_payment_link      │       │
│                     │            └──────────────────────────┘       │
│                     ▼                                               │
│            ┌─────────────────┐                                      │
│            │   humanizer     │  ← Make response natural             │
│            └────────┬────────┘                                      │
│                     │                                               │
│                     ▼                                               │
│            ┌─────────────────┐                                      │
│            │  send_response  │  ← Wazzup API + log to DB            │
│            └────────┬────────┘                                      │
│                     │                                               │
│               ┌─────┴─────┐                                         │
│               │ escalate? │                                         │
│               └─────┬─────┘                                         │
│                 ┌───┴───┐                                           │
│               yes      no                                           │
│                 │       │                                           │
│                 ▼       ▼                                           │
│  ┌──────────────────────┐                                           │
│  │  handle_escalation   │  ← Telegram + AmoCRM                      │
│  └──────────┬───────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│            END                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Структура проекта

```
fermer-langgraph/
├── src/
│   ├── __init__.py
│   ├── graph.py         # Главный граф LangGraph
│   ├── tools.py         # Tools для AI агента
│   ├── prompts.py       # System/User промпты
│   ├── integrations.py  # Внешние API (GraphQL, Wazzup, Telegram)
│   └── server.py        # FastAPI сервер
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
cp .env.example .env
# Заполните .env реальными токенами
```

### 3. Запуск сервера

```bash
cd src
python server.py
```

Сервер запустится на `http://localhost:8000`

## 📡 API Endpoints

### Webhook для Wazzup

```bash
POST /webhook/wazzup
```

Формат payload:
```json
{
  "messages": [{
    "chatId": "77001234567",
    "channelId": "xxx-xxx-xxx",
    "text": "Привет!",
    "status": "inbound",
    "chatType": "whatsapp"
  }]
}
```

### Прямой вызов

```bash
POST /process
```

```json
{
  "chat_id": "77001234567",
  "sender_id": "77001234567",
  "message": "Привет!",
  "source": "whatsapp",
  "channel_id": "xxx-xxx-xxx"
}
```

### Получение состояния

```bash
GET /graph/state/{chat_id}
GET /graph/history/{chat_id}
```

## 🔧 Миграция с n8n

### Соответствие нод

| n8n Node | LangGraph |
|----------|-----------|
| Webhook Wazzup | `POST /webhook/wazzup` |
| Extract Message Data | `extract_message_data()` |
| get fermer data (GraphQL) | `fetch_fermer_data_node()` |
| Simplify data | Включено в `fetch_fermer_data_node()` |
| Switch (triggers) | `select_trigger_type()` |
| set * prompt | `build_prompts()` |
| AI agent RAG | `ai_agent_node()` |
| Huminize Agent | `humanizer_node()` |
| Отправить сообщение | `send_response_node()` |
| If Human Needed | `should_escalate()` |
| Telegram/Notion | `handle_escalation_node()` |

### Соответствие Tools

| n8n Tool | LangGraph Tool |
|----------|----------------|
| get_schedule_by_club1 | `get_schedule_by_club()` |
| Fermer vector store | `search_knowledge_base()` |
| get_payment_link1 | `get_payment_link()` |

## 🔑 Преимущества LangGraph vs n8n

1. **Type Safety**: Pydantic модели для state и output
2. **Checkpointing**: Встроенное сохранение состояния
3. **Debugging**: Полная история состояний
4. **Testing**: Легко тестировать изолированно
5. **Версионирование**: Код в Git
6. **Масштабирование**: Легко добавлять ноды и условия

## 📊 State Schema

```python
class FermerState(TypedDict):
    # Input
    chat_id: str
    sender_id: str
    message: str
    source: str
    channel_id: str
    
    # Fermer Data
    user_id: str
    user_data: dict
    triggers: dict
    messages_history: list[str]
    
    # Prompts
    trigger_type: str
    system_prompt: str
    user_prompt: str
    
    # AI Output
    response_text: str
    humanized_response: str
    escalation_needed: bool
    escalation_reason: str
```

## 🧪 Тестирование

```bash
# Запуск тестов
pytest tests/ -v

# Тест конкретного сценария
pytest tests/test_first_training.py -v
```

## 📝 Примеры использования

### Python

```python
from src.graph import process_message

result = await process_message(
    chat_id="77001234567",
    sender_id="77001234567",
    message="Когда следующая тренировка?",
    source="whatsapp",
    channel_id="xxx",
)

print(result["response_text"])
```

### cURL

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "77001234567",
    "sender_id": "77001234567",
    "message": "Когда следующая тренировка?"
  }'
```

## 🔐 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API ключ |
| `HJ_AUTH_TOKEN` | JWT токен для Hero's Journey API |
| `PINECONE_API_KEY` | Pinecone API ключ |
| `WAZZUP_TOKEN` | Wazzup API токен |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot токен |
| `AMOCRM_TOKEN` | AmoCRM OAuth токен |

## 📚 Документация

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Docs](https://python.langchain.com/)
- [Pinecone Docs](https://docs.pinecone.io/)
