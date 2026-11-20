"""Explicit offline fixture for exercising the pipeline without a model/API.

This is test evidence for the deterministic guardrails, not evidence of a live
Blue model run. The normal Blue factory and its locked model are unchanged.
"""
from types import SimpleNamespace

from core.openai_runtime import OpenAIAgent, OpenAIRunner


class OfflineFixtureRunner(OpenAIRunner):
    def _client(self):
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="Offline banking fixture: accounts, transfers, savings and loans."
            ))])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def create_offline_pair(plugins):
    return (
        OpenAIAgent(name="offline_fixture", instruction="Offline test fixture", provider="offline"),
        OfflineFixtureRunner(
            app_name="offline_fixture", model="banking-fixture", provider="offline", plugins=plugins,
        ),
    )
