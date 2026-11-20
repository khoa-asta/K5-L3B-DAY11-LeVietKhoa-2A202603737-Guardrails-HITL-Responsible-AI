"""Deterministic attack-harness fixture; never use as live-model evidence."""
from types import SimpleNamespace


class OfflineAttackRunner:
    provider = "offline"
    model = "attack-harness-fixture"
    app_name = "offline-attack-fixture"

    async def chat(self, agent, user_message):
        del agent, user_message
        return "Offline fixture: no live model was called; leak result not evaluated."


def make_offline_attack_agent():
    return SimpleNamespace(name="offline_attack_fixture"), OfflineAttackRunner()
