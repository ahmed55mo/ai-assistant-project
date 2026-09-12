from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from backend.agents.langgraph_agent import LangGraphAgent
from backend.documents.chunking import DocumentChunk
from backend.rag.embeddings import LocalEmbeddingService
from backend.rag.errors import RetrievalError
from backend.rag.retrieval import RetrievalService
from backend.rag.tool import build_knowledge_search_tool
from backend.services.memory_service import ConversationMemory
from backend.tools.base import ToolRegistry
from backend.tools.langchain_tools import build_langchain_tools
from backend.tools.registry import build_registry
from backend.rag.vector_store import FaissVectorStore


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        user_id="user-1",
        conversation_id="conversation-1",
        chunk_index=0,
        text="Employees receive twenty annual leave days.",
        page_number=4,
        filename="company_policy.pdf",
    )


def _retrieval_service(conversation_id: str = "conversation-1") -> RetrievalService:
    embedding = LocalEmbeddingService(dimension=32)
    store = FaissVectorStore(embedding.dimension)
    chunk = _chunk()
    chunk = DocumentChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        user_id=chunk.user_id,
        conversation_id=conversation_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        page_number=chunk.page_number,
        filename=chunk.filename,
    )
    store.add([chunk], embedding.embed_texts([chunk.text]))
    return RetrievalService(embedding, store)


def _rag_registry(service: RetrievalService) -> ToolRegistry:
    return ToolRegistry([build_knowledge_search_tool(service)])


class ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Any]] = []

    def bind_tools(self, tools: list[Any]) -> "ScriptedModel":
        self.tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        return self.responses.pop(0)


def _call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge",
                "args": {
                    "query": "Employees receive twenty annual leave days."
                },
                "id": "knowledge-call-1",
                "type": "tool_call",
            }
        ],
    )


def test_rag_tool_is_registered_read_only_and_has_no_secrets() -> None:
    tool = build_langchain_tools(_rag_registry(_retrieval_service()))[0]

    assert tool.name == "search_knowledge"
    assert tool.metadata["read_only"] is True
    assert tool.metadata["requires_confirmation"] is False
    assert set(tool.args_schema.model_fields) == {
        "query",
        "document_id",
        "top_k",
        "score_threshold",
    }
    schema = json.dumps(tool.args_schema.model_json_schema())
    assert "user_id" not in schema
    assert "conversation_id" not in schema


def test_standard_registry_includes_knowledge_search_when_configured() -> None:
    names = set(build_registry(_retrieval_service()).names())

    assert "search_knowledge" in names


def test_rag_tool_returns_structured_results_and_delimited_context() -> None:
    tool = build_langchain_tools(_rag_registry(_retrieval_service()))[0]
    from backend.rag.tool import reset_trusted_scope, set_trusted_scope

    token = set_trusted_scope("user-1", "conversation-1")
    try:
        result = tool.invoke({"query": "annual leave"})
    finally:
        reset_trusted_scope(token)

    assert result["found"] is True
    assert result["sources"] == [
        {
            "document_id": "doc-1",
            "filename": "company_policy.pdf",
            "page_number": 4,
            "chunk_id": "chunk-1",
        }
    ]
    assert result["results"][0] == {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "filename": "company_policy.pdf",
        "page_number": 4,
        "chunk_index": 0,
        "text": "Employees receive twenty annual leave days.",
        "similarity_score": pytest.approx(result["results"][0]["similarity_score"]),
    }
    assert result["context"].startswith("<retrieved_knowledge>")
    assert "<document_content>" in result["context"]
    assert "</retrieved_knowledge>" in result["context"]


def test_rag_tool_no_results_are_explicit() -> None:
    service = RetrievalService(
        LocalEmbeddingService(dimension=8),
        FaissVectorStore(8),
    )
    tool = build_langchain_tools(_rag_registry(service))[0]
    from backend.rag.tool import reset_trusted_scope, set_trusted_scope

    token = set_trusted_scope("user-1", "conversation-1")
    try:
        result = tool.invoke({"query": "missing"})
    finally:
        reset_trusted_scope(token)

    assert result["found"] is False
    assert result["results"] == []
    assert result["sources"] == []
    assert "No relevant knowledge found." in result["context"]


def test_rag_tool_uses_trusted_scope_and_ignores_scope_arguments() -> None:
    retrieval = __import__("unittest.mock", fromlist=["Mock"]).Mock()
    retrieval.retrieve.return_value = []
    tool = build_langchain_tools(_rag_registry(retrieval))[0]
    from backend.rag.tool import reset_trusted_scope, set_trusted_scope

    token = set_trusted_scope("trusted-user", "trusted-conversation")
    try:
        tool.invoke({"query": "query"})
    finally:
        reset_trusted_scope(token)

    request = retrieval.retrieve.call_args.args[0]
    assert request.user_id == "trusted-user"
    assert request.conversation_id == "trusted-conversation"


def test_rag_tool_errors_are_controlled() -> None:
    retrieval = __import__("unittest.mock", fromlist=["Mock"]).Mock()
    retrieval.retrieve.side_effect = RetrievalError("internal details")
    tool = build_langchain_tools(_rag_registry(retrieval))[0]
    from backend.rag.tool import reset_trusted_scope, set_trusted_scope

    token = set_trusted_scope("user-1", "conversation-1")
    try:
        with pytest.raises(Exception, match="could not be completed"):
            tool.invoke({"query": "query"})
    finally:
        reset_trusted_scope(token)


def test_langgraph_calls_rag_then_generates_one_final_answer_without_confirmation() -> None:
    memory = ConversationMemory()
    conversation_id = memory.create_conversation()
    model = ScriptedModel([_call(), AIMessage(content="According to the policy, 20 days.")])
    agent = LangGraphAgent(
        memory,
        _rag_registry(_retrieval_service(conversation_id)),
        model=model,
    )
    agent.set_user_context("user-1")

    response = agent.respond(conversation_id, "How much annual leave do employees get?")

    assert response == "According to the policy, 20 days."
    assert len(model.calls) == 2
    assert conversation_id not in agent.pending
    tool_messages = [
        message for message in memory.get_history(conversation_id)
        if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "knowledge-call-1"
    assert "company_policy.pdf" in tool_messages[0]["content"]


def test_langgraph_source_metadata_reaches_structured_response() -> None:
    memory = ConversationMemory()
    conversation_id = memory.create_conversation()
    model = ScriptedModel([_call(), AIMessage(content="The policy says 20 days.")])
    agent = LangGraphAgent(
        memory,
        _rag_registry(_retrieval_service(conversation_id)),
        model=model,
    )
    agent.set_user_context("user-1")

    result = agent.respond_with_sources(conversation_id, "How much leave?")

    assert result == {
        "response": "The policy says 20 days.",
        "sources": [
            {
                "document_id": "doc-1",
                "filename": "company_policy.pdf",
                "page_number": 4,
                "chunk_id": "chunk-1",
            }
        ],
    }
    assert len(model.calls) == 2


def test_normal_response_does_not_call_rag_unnecessarily() -> None:
    memory = ConversationMemory()
    model = ScriptedModel([AIMessage(content="2 + 2 is 4.")])
    agent = LangGraphAgent(memory, _rag_registry(_retrieval_service()), model=model)
    conversation_id = memory.create_conversation()

    assert agent.respond(conversation_id, "What is 2 + 2?") == "2 + 2 is 4."
    assert len(model.calls) == 1


def test_rag_does_not_create_confirmation_state() -> None:
    memory = ConversationMemory()
    model = ScriptedModel([_call(), AIMessage(content="The policy says 20 days.")])
    agent = LangGraphAgent(memory, _rag_registry(_retrieval_service()), model=model)
    conversation_id = memory.create_conversation()

    agent.respond(conversation_id, "According to the policy, how much leave?")

    assert agent.pending == {}
