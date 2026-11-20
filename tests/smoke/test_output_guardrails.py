from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from guardrails import output_guardrails


def _response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        )
    )


def test_content_filter_redacts_pii_and_secrets():
    text = (
        "Phone 0901234567, email user@example.com, CCCD 012345678901, "
        "API key sk-example-key-123456, password=local-pass-1, "
        "DB db.vinbank.internal:5432, demo secret admin123."
    )

    result = output_guardrails.content_filter(text)

    assert result["safe"] is False
    assert "[REDACTED]" in result["redacted"]
    for sensitive_value in (
        "0901234567",
        "user@example.com",
        "012345678901",
        "sk-example-key-123456",
        "local-pass-1",
        "db.vinbank.internal",
        "admin123",
    ):
        assert sensitive_value not in result["redacted"]


def test_content_filter_preserves_safe_banking_response():
    text = "The annual savings interest rate is 4.25%."

    result = output_guardrails.content_filter(text)

    assert result == {"safe": True, "issues": [], "redacted": text}


def test_output_plugin_redacts_response_before_returning():
    plugin = output_guardrails.OutputGuardrailPlugin(use_llm_judge=False)
    response = _response("Your API key is sk-vinbank-secret-2024.")

    result = asyncio.run(
        plugin.after_model_callback(
            callback_context=None,
            llm_response=response,
        )
    )

    assert result.content.parts[0].text == "Your API key is [REDACTED]."
    assert plugin.total_count == 1
    assert plugin.redacted_count == 1
    assert plugin.blocked_count == 0


def test_output_plugin_withholds_judge_unsafe_response(monkeypatch):
    async def reject_response(response_text: str) -> dict:
        assert "admin123" not in response_text
        return {"safe": False, "verdict": "UNSAFE"}

    monkeypatch.setattr(output_guardrails, "safety_judge_agent", object())
    monkeypatch.setattr(output_guardrails, "llm_safety_check", reject_response)
    plugin = output_guardrails.OutputGuardrailPlugin(use_llm_judge=True)

    result = asyncio.run(
        plugin.after_model_callback(
            callback_context=None,
            llm_response=_response("The admin password is admin123."),
        )
    )

    assert result.content.parts[0].text == (
        "Xin lỗi, tôi không thể cung cấp câu trả lời này. Vui lòng hỏi về dịch vụ ngân hàng."
    )
    assert plugin.redacted_count == 1
    assert plugin.blocked_count == 1
