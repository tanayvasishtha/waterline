"""Bedrock-backed chooser: a second opinion inside the deterministic loop.

The LLM sees the same evidence the rules see (metrics, what was tried) and picks
one action plus a one-line reason. Any failure -- no model configured, no network,
a malformed answer, a budget cap -- falls back to the deterministic policy, which
is the floor we always stand on. The LLM never sees the gage value either.
"""

import json
from datetime import date

import boto3
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

from waterline.models import AgentAction
from waterline.pipeline import FrameResult
from waterline.settings import Settings

from .policy import Decision, LoopState, decide

SYSTEM = """You are the action policy of a river-gauge camera agent.
Given metrics about one frame and the actions already tried, choose the single
best next action to produce a confident water-stage reading, or a terminal
decision. Reply with JSON only: {"action": "<action>", "reason": "<one line>"}.
Valid actions: enhance_clahe, enhance_gamma, mask_glare, alt_reference,
fetch_prev, fetch_next, accept, abstain, human_review.
Never guess a stage value. Prefer accept when sigma is small and methods agree.
Prefer fetch_next for night or unreadable frames. Escalate only when stuck."""


def _payload(result: FrameResult, state: LoopState, settings: Settings) -> dict[str, object]:
    q, r = result.quality, result.reading
    return {
        "registered": result.registration is not None,
        "night": q.is_night,
        "blur_var": round(q.blur_var, 1),
        "glare_frac": round(q.glare_frac, 3),
        "plate_visibility": round(q.plate_visibility, 1),
        "stage_ft": r.stage_ft,
        "sigma_ft": r.sigma_ft,
        "methods_disagree": r.methods_disagree,
        "reason": r.reason,
        "tried": sorted(a.value for a in state.tried),
        "steps_left": settings.agent_step_budget - state.steps,
    }


class BedrockChooser:
    """Calls the configured Bedrock model; degrades to `decide` on any problem."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: BedrockRuntimeClient | None = None
        self._calls_today = 0
        self._today = date.today()
        self.used_llm = False

    def _reset_if_new_day(self) -> None:
        if date.today() != self._today:
            self._today = date.today()
            self._calls_today = 0

    def _client_lazy(self) -> BedrockRuntimeClient:
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name="us-east-1")
        return self._client

    def _suggest(self, result: FrameResult, state: LoopState) -> Decision | None:
        self._reset_if_new_day()
        if (
            not self.settings.bedrock_model_id
            or self._calls_today >= self.settings.bedrock_daily_cap
        ):
            return None
        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.settings.bedrock_max_tokens,
                "system": SYSTEM,
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(_payload(result, state, self.settings)),
                    }
                ],
            }
            response = self._client_lazy().invoke_model(
                modelId=self.settings.bedrock_model_id, body=json.dumps(body)
            )
            self._calls_today += 1
            self.used_llm = True
            text = response["body"].read().decode()
            outer = json.loads(text)
            content = outer["content"][0]["text"]
            answer = json.loads(content)
            action = AgentAction(answer["action"])
            reason = str(answer["reason"])[:200]
            if action in state.tried:
                return None
            return Decision(action, f"llm: {reason}")
        except Exception:  # never trust the call; the rules are the floor
            return None

    def __call__(self, result: FrameResult, state: LoopState, settings: Settings) -> Decision:
        suggested = self._suggest(result, state)
        return suggested if suggested is not None else decide(result, state, settings)
