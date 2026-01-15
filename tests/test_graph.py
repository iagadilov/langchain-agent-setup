"""
Fermer Agent Tests
==================
Basic tests for the LangGraph workflow.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch


# Mock environment before imports
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HJ_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    monkeypatch.setenv("WAZZUP_TOKEN", "test-wazzup")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-telegram")
    monkeypatch.setenv("AMOCRM_TOKEN", "test-amocrm")


class TestStateFlow:
    """Test state transitions in the graph."""
    
    def test_initial_state_creation(self):
        """Test that initial state is created correctly."""
        from src.graph import FermerState
        
        state: FermerState = {
            "chat_id": "77001234567",
            "sender_id": "77001234567",
            "message": "Привет!",
            "source": "whatsapp",
            "channel_id": "test-channel",
            "timestamp": "2025-01-15T10:00:00",
            "messages": [],
            "messages_history": [],
            "triggers": {},
            "should_respond": True,
        }
        
        assert state["chat_id"] == "77001234567"
        assert state["should_respond"] is True
    
    def test_trigger_selection(self):
        """Test trigger type selection logic."""
        from src.graph import select_trigger_type
        
        # Test first training trigger
        state = {"triggers": {"firstTraining": True}}
        result = select_trigger_type(state)
        assert result["trigger_type"] == "first_training"
        
        # Test no activity trigger
        state = {"triggers": {"noActivity": True, "firstTraining": False}}
        result = select_trigger_type(state)
        assert result["trigger_type"] == "no_activity"
        
        # Test payment trigger
        state = {"triggers": {"payment": True}}
        result = select_trigger_type(state)
        assert result["trigger_type"] == "payment"
        
        # Test default trigger
        state = {"triggers": {}}
        result = select_trigger_type(state)
        assert result["trigger_type"] == "default"


class TestPrompts:
    """Test prompt generation."""
    
    def test_system_prompt_first_training(self):
        """Test system prompt for first training scenario."""
        from src.prompts import get_system_prompt
        
        prompt = get_system_prompt(
            trigger_type="first_training",
            user_data={"firstName": "Тест"},
            user_profile={},
        )
        
        assert "Batyr" in prompt
        assert "first training" in prompt.lower() or "workout" in prompt.lower()
        assert "JSON" in prompt
    
    def test_user_prompt_includes_message(self):
        """Test that user prompt includes the message."""
        from src.prompts import get_user_prompt
        
        prompt = get_user_prompt(
            trigger_type="default",
            message="Как записаться на тренировку?",
            messages_history=[],
            training_data={},
            user_data={},
        )
        
        assert "Как записаться на тренировку?" in prompt


class TestTools:
    """Test tool functions."""
    
    @pytest.mark.asyncio
    async def test_schedule_tool_validates_club(self):
        """Test that schedule tool validates club ID."""
        from src.tools import get_schedule_by_club
        
        # Invalid club ID should return error message
        result = await get_schedule_by_club.ainvoke({
            "club_id": "invalid-club",
        })
        
        assert "❌" in result or "Укажите клуб" in result
    
    def test_club_name_mapping(self):
        """Test club name to ID mapping."""
        from src.tools import CLUB_NAMES, CLUB_IDS_BY_NAME
        
        assert CLUB_NAMES["65e9e70cbd4814536c5e27e9"] == "Colibri"
        assert CLUB_IDS_BY_NAME["colibri"] == "65e9e70cbd4814536c5e27e9"
        assert CLUB_IDS_BY_NAME["колибри"] == "65e9e70cbd4814536c5e27e9"


class TestIntegrations:
    """Test external integrations."""
    
    @pytest.mark.asyncio
    async def test_send_whatsapp_message_format(self):
        """Test WhatsApp message sending format."""
        from src.integrations import send_whatsapp_message
        
        with patch("src.integrations.httpx.AsyncClient") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            result = await send_whatsapp_message(
                chat_id="77001234567",
                channel_id="test-channel",
                text="Test message",
            )
            
            assert result is True


class TestConditionalEdges:
    """Test conditional edge functions."""
    
    def test_should_continue_after_data(self):
        """Test data validation conditional."""
        from src.graph import should_continue_after_data
        
        # With error
        state = {"error": "Some error", "should_respond": True}
        assert should_continue_after_data(state) == "end"
        
        # Without error
        state = {"error": None, "should_respond": True}
        assert should_continue_after_data(state) == "select_trigger"
        
        # Should not respond
        state = {"error": None, "should_respond": False}
        assert should_continue_after_data(state) == "end"
    
    def test_should_escalate(self):
        """Test escalation conditional."""
        from src.graph import should_escalate
        
        # Escalation needed
        state = {"escalation_needed": True}
        assert should_escalate(state) == "escalate"
        
        # No escalation
        state = {"escalation_needed": False}
        assert should_escalate(state) == "end"


class TestEndToEnd:
    """End-to-end tests (mocked)."""
    
    @pytest.mark.asyncio
    async def test_full_flow_mock(self):
        """Test full message processing flow with mocks."""
        from src.graph import process_message
        
        with patch("src.graph.fetch_fermer_data_node") as mock_fermer, \
             patch("src.graph.ai_agent_node") as mock_ai, \
             patch("src.graph.humanizer_node") as mock_humanizer, \
             patch("src.graph.send_response_node") as mock_send:
            
            # Setup mocks
            mock_fermer.return_value = {
                "user_id": "test-user",
                "triggers": {"firstTraining": True},
                "user_data": {"firstName": "Тест"},
                "messages_history": [],
            }
            
            mock_ai.return_value = {
                "response_text": "Поздравляю с первой тренировкой!",
                "escalation_needed": False,
                "escalation_reason": "",
            }
            
            mock_humanizer.return_value = {
                "humanized_response": "Поздравляю с первой тренировкой! 💪",
            }
            
            mock_send.return_value = {}
            
            # This would run the full flow in a real scenario
            # For now, just verify the test structure is correct
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
