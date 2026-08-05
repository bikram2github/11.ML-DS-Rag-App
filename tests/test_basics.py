from unittest.mock import patch
from backend import get_chatbot

@patch("backend.get_rag_chain")
def test_chatbot_initializes(mock_rag):
    mock_rag.return_value = lambda x: "mock response"
    chatbot = get_chatbot()
    assert chatbot is not None
